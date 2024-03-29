import numpy as np
import torch as t
import torch
import torchaudio
import museval
import os
# import wandb
from pathlib import Path
from tqdm import tqdm
from jukebox.hparams import setup_hparams
from jukebox.make_models import make_vqvae, make_prior
from jukebox.transformer.ops import filter_logits
from jukebox.utils.dist_utils import setup_dist_from_mpi
from jukebox.utils.logger import def_tqdm
from jukebox.utils.sample_utils import get_starts
from jukebox.utils.torch_utils import empty_cache
from jukebox.data.files_dataset import FilesAudioDataset
from torch.utils.data.distributed import DistributedSampler
from jukebox.utils.audio_utils import audio_preprocess, audio_postprocess
from torch.utils.data import DataLoader, Dataset, BatchSampler, RandomSampler
from datasets import ChunkedPairsDataset
from utils import get_raw_to_tokens 
from diba_interfaces import SparseLikelihood
from utils import _compute_log_posterior
rank, local_rank, device = setup_dist_from_mpi(port=29524)
from timeit import default_timer as timer
from typing import Tuple

def make_models(vqvae_path, priors_list, sample_length, downs_t, sample_rate,
                levels=3, level=2, fp16=True, device='cuda'):
    # construct openai vqvae and priors
    vqvae = make_vqvae(setup_hparams('vqvae', dict(sample_length=sample_length, downs_t=downs_t, sr=sample_rate,
                                                   restore_vqvae=vqvae_path)), device)
    prior_path_0 = priors_list[0]
    prior_path_1 = priors_list[1]

    prior_0 = make_prior(setup_hparams('small_prior', dict(levels=levels, level=level, labels=None,
                                                           restore_prior=prior_path_0, c_res=1, fp16_params=fp16,
                                                           n_ctx=8192)), vqvae, device)
    prior_1 = make_prior(setup_hparams('small_prior', dict(levels=levels, level=level, labels=None,
                                                           restore_prior=prior_path_1, c_res=1, fp16_params=fp16,
                                                           n_ctx=8192)), vqvae, device)
    priors = [prior_0, prior_1]
    return vqvae, priors


def create_mixture_from_audio_files(m1, m2, raw_to_tokens, sample_tokens,
                                    vqvae, save_path, sample_rate, alpha, device='cuda', shift=0.):

    # controllare che m1 e m2 devono essere di dimensioni (1, length) es (1, 5060608)

    shift = int(shift * sample_rate)
    assert sample_tokens * raw_to_tokens <= min(m1.shape[-1], m2.shape[-1]), "Sources must be longer than sample_tokens"
    minin = sample_tokens * raw_to_tokens
    m1_real    = m1[:, shift:shift+minin]
    m2_real    = m2[:, shift:shift+minin]
    mix_real       = alpha[0]*m1_real + alpha[1]*m2_real
 
    z_m1 = vqvae.encode(m1_real.unsqueeze(-1).to(device), start_level=2, bs_chunks=1)[0]
    z_m2 = vqvae.encode(m2_real.unsqueeze(-1).to(device), start_level=2, bs_chunks=1)[0]
    z_mixture = vqvae.encode(mix_real.unsqueeze(-1).to(device), start_level=2, bs_chunks=1)[0]

    latent_mix = vqvae.bottleneck.decode([z_mixture]*3)[-1]

    mix = vqvae.decode([z_mixture], start_level=2, bs_chunks=1).squeeze(-1)  # 1, 8192*128
    m1 = vqvae.decode([z_m1], start_level=2, bs_chunks=1).squeeze(-1)  # 1, 8192*128
    m2 = vqvae.decode([z_m2], start_level=2, bs_chunks=1).squeeze(-1)  # 1, 8192*128

    return mix, latent_mix, z_mixture, m1, m2, m1_real, m2_real, mix_real

def sample_level(vqvae, priors, likelihood, z_mixture, m, n_samples, n_ctx, hop_length, sample_tokens, sigma=0.01, context=8, fp16=False, temp=1.0,
                 alpha=None, top_k=0, top_p=0.0, bs_chunks=1, window_mode='constant', l_bins=2048, raw_to_tokens=128, device=None,
                 chunk_size=None, latent_loss=True, top_k_posterior=0, delta_likelihood=False):
    xs_0 = torch.zeros(n_samples, 0, dtype=torch.long, device=device)
    xs_1 = torch.zeros(n_samples, 0, dtype=torch.long, device=device)

    if sample_tokens >= n_ctx:
        for start in get_starts(sample_tokens, n_ctx, hop_length):
            xs_0, xs_1, log_p_0_sum, log_p_1_sum, log_likelihood_sum = sample_single_window(xs_0, xs_1, vqvae, priors, m, n_samples, n_ctx, start=start, sigma=sigma,
                                                                                            context=context, fp16=fp16, temp=temp, alpha=alpha, top_k=top_k,
                                                                                            top_p=top_p, bs_chunks=bs_chunks, window_mode=window_mode, l_bins=l_bins,
                                                                                            raw_to_tokens=raw_to_tokens, device=device, chunk_size=chunk_size,
                                                                                            latent_loss=latent_loss, top_k_posterior=top_k_posterior, delta_likelihood=delta_likelihood)
    else:
        xs_0, xs_1, log_p_0_sum, log_p_1_sum, log_likelihood_sum = ancestral_sample(vqvae, priors, likelihood, z_mixture, m, n_samples, sample_tokens=sample_tokens, sigma=sigma,
                                                                                    context=context, fp16=fp16, temp=temp, alpha=alpha, top_k=top_k, top_p=top_p,
                                                                                    bs_chunks=bs_chunks, window_mode=window_mode, l_bins=l_bins,
                                                                                    raw_to_tokens=raw_to_tokens, device=device, latent_loss=latent_loss,
                                                                                    top_k_posterior=top_k_posterior, delta_likelihood=delta_likelihood)
    return xs_0, xs_1, log_p_0_sum, log_p_1_sum, log_likelihood_sum

# log_p_0_sum, log_p_1_sum, res_0, res_1, remaining_0, remaining_1, mix, alpha, bs, rejection_sigma=rejection_sigma, n_samples=sample_tokens
def rejection_sampling(nll0, nll1, res0, res1, remaining0, remaining1, m, alpha, bs, rejection_sigma, n_samples):
    nll_sum_0_sorted, indices_nll_sum_0_sorted = torch.sort(nll0)
    nll_sum_1_sorted, indices_nll_sum_1_sorted = torch.sort(nll1)

    global_likelihood = torch.zeros((bs, bs))
    global_posterior = torch.zeros((bs, bs))
    global_prior = torch.zeros((bs, bs))
    global_l2 = torch.zeros((bs, bs))

    # sdr(alpha[0]*res0[i, :].reshape(1, -1, 1).cpu().numpy() +
    #                             alpha[1]*res1[j, :].reshape(1, -1, 1).cpu().numpy(),
    #                            m.unsqueeze(-1).cpu().numpy())

    for i in tqdm(range(bs), desc="Rejection sampling"):
        for j in range(bs):
            # drop = remaining0[i] and remaining1[j] #-np.inf if (not remaining0[i] or not remaining1[j]) else 0.

            global_prior[i, j] = (nll0[i] + nll1[j]) / n_samples # if drop else -np.inf
            global_l2[i, j] = torch.linalg.norm((alpha[0]*res0[i] +
                                                 alpha[1]*res1[j] - m), dim=-1)**2

    mp = global_prior.mean()
    ml = global_l2.mean()
    sigma_rejection_squared = - ml / (2*mp)
    # print(f"sigma_rejection = {torch.sqrt(sigma_rejection_squared)}")
    global_likelihood = (-(1/(2.*(sigma_rejection_squared)))* global_l2)
    #if drop else 0.)
    # global_posterior[i, j] = global_likelihood[i, j] + global_prior[i, j]

    global_prior = global_prior.reshape(bs*bs)
    global_prior = torch.distributions.Categorical(logits=global_prior)
    global_prior_p = global_prior.probs.reshape(bs, bs)
    global_prior = global_prior.logits.reshape(bs, bs)

    global_posterior = global_prior + global_likelihood

    global_posterior = global_posterior.reshape(bs*bs) # n_samples, 2048 * 2048
    global_posterior = torch.distributions.Categorical(logits=global_posterior)
    global_posterior = global_posterior.probs.reshape(bs, bs)

    marginal_0 = global_posterior.sum(dim=-1)
    marginal_1 = global_posterior.sum(dim=0)
    marginal_1_sorted, marginal_1_idx_sorted = torch.sort(marginal_1)
    marginal_0_sorted, marginal_0_idx_sorted = torch.sort(marginal_0)

    #print(f"marginal_0_sorted = {marginal_0_sorted}")
    #print(f"marginal_1_sorted = {marginal_1_sorted}")
    #print(f"marginal_0_idx_sorted = {marginal_0_idx_sorted}")
    #print(f"marginal_1_idx_sorted = {marginal_1_idx_sorted}")

    global_l2_vectorized = global_l2.reshape(bs*bs)  # righe: prior_0, colonne: prior_1
    global_l2_topk_vectorized, global_l2_topk_idx_vectorized  = torch.topk(global_l2_vectorized, k=10, largest=False)
    #print(f"global_l2_topk = {global_l2_topk_vectorized}")
    #print(f"global_l2_topk_idx = {[(idx // bs, idx % bs) for idx in global_l2_topk_idx_vectorized]}")

    rejection_index = torch.argmax(global_posterior)
    return (marginal_0_sorted, marginal_1_sorted, marginal_0_idx_sorted, marginal_1_idx_sorted, rejection_index // bs, rejection_index % bs)



def evaluate_sdr(gt0, gt1, res0, res1):
    sdr_0 = torch.zeros((res0.shape[0],))
    sdr_1 = torch.zeros((res1.shape[0],))
    for i in range(res0.shape[0]):
        sdr_0[i] = sdr(gt0.unsqueeze(-1).cpu().numpy(), res0[i, :].reshape(1, -1, 1).cpu().numpy())
        sdr_1[i] = sdr(gt1.unsqueeze(-1).cpu().numpy(), res1[i, :].reshape(1, -1, 1).cpu().numpy())

    sdr_0_sorted, sdr_0_sorted_idx = torch.sort(sdr_0)
    sdr_1_sorted, sdr_1_sorted_idx = torch.sort(sdr_1)
    return sdr_0, sdr_1, sdr_0_sorted, sdr_1_sorted, sdr_0_sorted_idx, sdr_1_sorted_idx


def sample_single_window(xs_0, xs_1, vqvae, priors, m, n_samples, n_ctx, start=0, sigma=0.01, context=8, fp16=False, temp=1.0,
                         alpha=None, top_k=0, top_p=0.0, bs_chunks=1, window_mode='constant', l_bins=2048, raw_to_tokens=128, device=None,
                         chunk_size=None, latent_loss=True, top_k_posterior=0, delta_likelihood=False):
    end = start + n_ctx
    # get z already sampled at current level
    x_0 = xs_0[:, start:end]
    x_1 = xs_1[:, start:end]

    sample_tokens = end - start
    conditioning_tokens, new_tokens = x_0.shape[1], sample_tokens - x_0.shape[1]

    if new_tokens <= 0:
        return xs_0, xs_1

    empty_cache()

    x_0, x_1, log_p_0_sum, log_p_1_sum, log_likelihood_sum = sample(x_0, x_1, vqvae, priors, m, n_samples, sample_tokens=sample_tokens, sigma=sigma, context=context,
                                                                    fp16=fp16, temp=temp, alpha=alpha, top_k=top_k, top_p=top_p, bs_chunks=bs_chunks,
                                                                    window_mode=window_mode, l_bins=l_bins, raw_to_tokens=raw_to_tokens, device=device,
                                                                    chunk_size=chunk_size, latent_loss=latent_loss, top_k_posterior=top_k_posterior, delta_likelihood=delta_likelihood)

    # Update z with new sample
    x_0_new = x_0[:, -new_tokens:]
    x_1_new = x_1[:, -new_tokens:]

    xs_0 = torch.cat([xs_0, x_0_new], dim=1)
    xs_1 = torch.cat([xs_1, x_1_new], dim=1)

    return xs_0, xs_1, log_p_0_sum, log_p_1_sum, log_likelihood_sum

def _sample(posterior_data: torch.Tensor, coords: torch.LongTensor) -> Tuple[torch.LongTensor, torch.LongTensor]:
    # check input shape
    batch_size, nnz_posterior = posterior_data.shape
    num_dims, nnz_coords = coords.shape

    assert num_dims == 2
    assert nnz_coords == nnz_posterior

    samples = torch.distributions.Categorical(logits=posterior_data).sample()
    x_0, x_1 = torch.gather(coords, dim=-1, index=samples.view(1, batch_size).repeat(num_dims, 1))
    return x_0, x_1

def ancestral_sample(vqvae, priors, likelihood, z_mixture, m, n_samples, sample_tokens, sigma=0.01, context=8, fp16=False, temp=1.0, alpha=None,
                     top_k=0, top_p=0.0, bs_chunks=1, window_mode='constant', l_bins=2048, raw_to_tokens=128, device=None,
                     latent_loss=True, top_k_posterior=0, delta_likelihood=False):
    
    x_cond = torch.zeros((n_samples, 1, priors[0].width), dtype=torch.float).to(device)
    xs_0, xs_1, x_0, x_1 = [], [], None, None
    
    log_post_sum = torch.zeros(1, 1, dtype=torch.long, device=device)

    log_p_0_sum = torch.zeros((n_samples,)).to(device)
    log_p_1_sum = torch.zeros((n_samples,)).to(device)
    log_likelihood_sum = torch.zeros((n_samples,)).to(device)


    for sample_t in def_tqdm(range(0, sample_tokens)):
        
        # get logits
        x_0, cond_0 = priors[0].get_emb(sample_t, n_samples, x_0, x_cond, y_cond=None)
        priors[0].transformer.check_cache(n_samples, sample_t, fp16)
        x_0 = priors[0].transformer(x_0, encoder_kv=None, sample=True, fp16=fp16) # Transformer
        if priors[0].add_cond_after_transformer:
            x_0 = x_0 + cond_0
        assert x_0.shape == (n_samples, 1, priors[0].width)
        x_0 = priors[0].x_out(x_0) # Predictions
        # end get logits

        x_0 = x_0 / temp
        x_0 = filter_logits(x_0, top_k=top_k, top_p=top_p)
        p_0 = torch.distributions.Categorical(logits=x_0).probs # Sample and replace x
        log_p_0 = torch.log(p_0) # n_samples, 1, 2048

        # get logits 
        x_1, cond_1 = priors[1].get_emb(sample_t, n_samples, x_1, x_cond, y_cond=None)
        priors[1].transformer.check_cache(n_samples, sample_t, fp16)
        x_1 = priors[1].transformer(x_1, encoder_kv=None, sample=True, fp16=fp16) # Transformer
        if priors[1].add_cond_after_transformer:
            x_1 = x_1 + cond_1
        x_1 = priors[1].x_out(x_1) # Predictions
        # end get logits
        
        x_1 = x_1 / temp
        x_1 = filter_logits(x_1, top_k=top_k, top_p=top_p)
        p_1 = torch.distributions.Categorical(logits=x_1).probs # Sample and replace x
        log_p_1 = torch.log(p_1) # n_samples, 1, 2048

        # NOTE: during token separation batch-size should be equal to num. samples
        assert len(log_p_0) == len(log_p_1) == n_samples
        assert log_p_0.shape[-1] == log_p_1.shape[-1] == likelihood.get_tokens_count()

      
        ### START LOG LIKELIHOOD ###

        #################### log likelihood in sparse COO format ####################
        ll_coords, ll_data = likelihood._get_log_likelihood(int(z_mixture[0, sample_t].item()))

        # compute log posterior
        if ll_coords.numel() > 0:
            # Note: posterior_data has shape (n_samples, nonzeros)   
            posterior_data = _compute_log_posterior(ll_data, ll_coords, log_p_0.view(-1, log_p_0.shape[-1]), log_p_1.view(-1, log_p_1.shape[-1]))

            x_0, x_1 = _sample(posterior_data, ll_coords)
        
        else:
        
            raise RuntimeError(f"Code {z_mixture[:, sample_t]} is not available in likelihood!")

        ##### END LIKELIHOOD #####

        # Note: x_0, x_1 have shape (n_sample, 1)
        x_0 = x_0.reshape(n_samples, -1)
        x_1 = x_1.reshape(n_samples, -1)

        log_p_0_sum += log_p_0[range(x_0.shape[0]), :, x_0.squeeze(-1)].squeeze(-1)
        log_p_1_sum += log_p_1[range(x_1.shape[0]), :, x_1.squeeze(-1)].squeeze(-1)
        #  log_likelihood_sum += log_likelihood[x_1, x_0].squeeze(-1)

        xs_0.append(x_0.clone())
        xs_1.append(x_1.clone())

    del x_0
    del x_1
    priors[0].transformer.del_cache()
    priors[1].transformer.del_cache()

    x_0 = torch.cat(xs_0, dim=1) # n_samples, sample_tokens
    #print(f"{x_0.shape = }")
    x_0 = priors[0].postprocess(x_0, sample_tokens) # n_samples, sample_tokens
    #print(f"{x_0.shape = }")
    x_1 = torch.cat(xs_1, dim=1)
    x_1 = priors[1].postprocess(x_1, sample_tokens)
    return x_0, x_1, log_p_0_sum, log_p_1_sum, None #log_likelihood_sum

def sdr(track1, track2):
    sdr_metric = museval.evaluate(track1, track2)
    sdr_metric[0][sdr_metric[0] == np.inf] = np.nan
    return np.nanmedian(sdr_metric[0])

def sample(xs_0, xs_1, vqvae, priors, m, n_samples, sample_tokens, sigma=0.01, context=8, fp16=False, temp=1.0,
           alpha=None, top_k=0, top_p=0.0, bs_chunks=1, window_mode='constant', l_bins=2048, raw_to_tokens=128,
           device=None, chunk_size=None, latent_loss=True, top_k_posterior=0, delta_likelihood=False):
    no_past_context = (xs_0 is None or xs_0.shape[1] == 0 or xs_1 is None or xs_1.shape[1] == 0)
    with torch.no_grad():
        if no_past_context:
            x_0, x_1, log_p_0_sum, log_p_1_sum, log_likelihood_sum = ancestral_sample(vqvae, priors, m, n_samples, sample_tokens=sample_tokens, sigma=sigma,
                                                                                      context=context, fp16=fp16, temp=temp, alpha=alpha, top_k=top_k, top_p=top_p,
                                                                                      bs_chunks=bs_chunks, window_mode=window_mode, l_bins=l_bins,
                                                                                      raw_to_tokens=raw_to_tokens, device=device, latent_loss=latent_loss,
                                                                                      top_k_posterior=top_k_posterior, delta_likelihood=delta_likelihood)
        else:
            #x_0, x_1 = primed_sample(xs_0, xs_1, vqvae, priors, m, n_samples, sample_tokens=sample_tokens,
            #                         sigma=sigma, context=context, fp16=fp16, temp=temp, alpha=alpha, top_k=top_k,
            #                         top_p=top_p, bs_chunks=bs_chunks, window_mode=window_mode, l_bins=l_bins,
            #                         raw_to_tokens=raw_to_tokens, device=device, chunk_size=chunk_size,
            #                         latent_loss=latent_loss, top_k_posterior=top_k_posterior, delta_likelihood=delta_likelihood)
            nll_sum_0 = None
            nll_sum_1 = None
            print("Error: no_past_context=False")
    return x_0, x_1, nll_sum_0, nll_sum_1, None


if __name__ == '__main__':

    raw_to_tokens = 64 #Downsampling factor
    l_bins = 2048
    sample_tokens = 1024 #How many tokens to sample Default: 1024 for ~ 3s.
    sample_length = raw_to_tokens * sample_tokens #131072?

    hps = setup_hparams("vqvae", {})
    downs_t = (2, 2, 2)
    commit = 1
    alpha = [0.5, 0.5]
    sample_rate = 22050
    min_duration = 11.90  
    levels = 3
    level = 2
    fp16 = True
    labels = False

    SILENCE_THRESHOLD = 1.5e-5


    drums_audio_files_dir= '../../test/bass'
    bass_audio_files_dir=  '../../test/drums' 

    aug_blend = False

    priors_list = ['./logs/prior_bass/checkpoint_latest.pth.tar',
                   './logs/prior_drums/checkpoint_latest.pth.tar']

    restore_vqvae = './logs/lq_vae/checkpoint_step_19160.pth.tar'
    sum_frequencies_path = "./logs/vqvae_sum_distribution/sum_dist_10000.npz"
    #vqvae = make_vqvae(setup_hparams('vqvae', dict(sample_length=sample_length, downs_t=downs_t, sr=sample_rate, commit=commit, restore_vqvae=restore_vqvae)), device)
    vqvae, priors = make_models(restore_vqvae, priors_list, sample_length, downs_t, sample_rate,
                                levels=levels, level=level, fp16=fp16, device=device)

    
    # setup dataset
    raw_to_tokens = get_raw_to_tokens(vqvae.strides_t, vqvae.downs_t)
    dataset = ChunkedPairsDataset(
        instrument_1_audio_dir=drums_audio_files_dir,
        instrument_2_audio_dir=bass_audio_files_dir,
        sample_rate=sample_rate,
        max_chunk_size=raw_to_tokens * sample_tokens,
        min_chunk_size=raw_to_tokens,
    )

    # get samples
    loader = DataLoader(dataset, batch_size=1, num_workers=0)
    sparse_likelihood = SparseLikelihood(sum_frequencies_path, device, 3.0)

    n_ctx = min(priors[0].n_ctx, priors[1].n_ctx)
    hop_length = n_ctx // 2
    #multi_sigma = torch.tensor([0.2,0.4,0.6,0.8]*8).cuda()
    sigma           = 0.4  # 316227766  #0.316227766 # 0.316227766 # was the best 0.004 both for filtered and none
    rejection_sigma = 0.06 #0.0625 #10000 #0.04 #0.02 #0.018  # 0. #0.02
    context         = 50   # quello che funziona: 10
    bs              = 64
    top_k_posterior = 0
    bs_chunks = 1
    chunk_size = 32
    window_mode = 'constant'
    latent_loss = True
    delta_likelihood = False
    save_path = Path('./script/results-test/')
    use_posterior_argmax = True

    shuffle_dataset = False
    np.random.seed(0)

    total_sdr0_gt_rejection_marginal, total_sdr1_gt_rejection_marginal = torch.tensor(0.).cuda(), torch.tensor(0.).cuda()
    total_sdr0_real_rejection_marginal, total_sdr1_real_rejection_marginal = torch.tensor(0.).cuda(), torch.tensor(0.).cuda()
    total_sdr0_gt, total_sdr1_gt = torch.tensor(0.).cuda(), torch.tensor(0.).cuda()
    total_sdr0_real, total_sdr1_real = torch.tensor(0.).cuda(), torch.tensor(0.).cuda()

    total_sdr0_gt_rejection_posterior, total_sdr1_gt_rejection_posterior = torch.tensor(0.).cuda(), torch.tensor(0.).cuda()
    total_sdr0_real_rejection_posterior, total_sdr1_real_rejection_posterior = torch.tensor(0.).cuda(), torch.tensor(0.).cuda()
    avg_time_sep = 0.
    avg_time_rej = 0.
    
    # wandb.init(
    #     id = "hybrid_evaluation", 
    #     resume="must",
    #     # set the wandb project where this run will be logged
    #     project="DL-Project-LQVAE",
    #     name="hybrid_evaluation",
    #     # track hyperparameters and run metadata
    #     config={
    #     "dataset": "Slakh2100-Drums&Bass",
    #     "test-size": 90,
    #     "sr": 22050,
    #     "levels": 3,
    #     "l_bins": 2048,
    #     "sample_length": sample_length,
    #     "raw_to_tokens":raw_to_tokens,
    #     "sample_tokens":sample_tokens,
    #     "sigma_rej": "not_zero", 
    #     }
    # )
        
    for batch_idx, batch in enumerate(loader):

        chunk_path = save_path / f"{batch_idx}"
        
        if chunk_path.exists():
            print(f"Skipping path: {chunk_path}")
            continue

        chunk_path.mkdir(parents=True)
        # load audio tracks
        orig1, orig2 = batch
        print(f"chunk {batch_idx+1} out of {len(dataset)}")

        start_sep = timer()

        # mix: E->D, latent_mix: code vecs, m0,m1: E->D, m0_real,m1_real: real
        mix, latent_mix, z_mixture, m0, m1, m0_real, m1_real, mix_real = create_mixture_from_audio_files(orig1.view(1,-1), orig2.view(1,-1),
                                                                                                raw_to_tokens,
                                                                                                sample_tokens,
                                                                                                vqvae,
                                                                                                save_path,
                                                                                                sample_rate,
                                                                                                alpha
                                                                                                )
        m = latent_mix

        # added z_mixture and likelihood to the function
        x_0, x_1, log_p_0_sum, log_p_1_sum, log_likelihood_sum = sample_level(vqvae,
                                                                                [priors[0].prior,
                                                                                priors[1].prior],
                                                                                m=m, z_mixture=z_mixture.cpu(), likelihood=sparse_likelihood, n_ctx=n_ctx,
                                                                                hop_length=hop_length,
                                                                                alpha=alpha,
                                                                                n_samples=bs,
                                                                                sample_tokens=sample_tokens,
                                                                                sigma=sigma,
                                                                                context=context,
                                                                                fp16=fp16,
                                                                                bs_chunks=bs_chunks,
                                                                                window_mode=window_mode,
                                                                                l_bins=l_bins,
                                                                                raw_to_tokens=raw_to_tokens,
                                                                                device=device,
                                                                                chunk_size=chunk_size,
                                                                                latent_loss=latent_loss,
                                                                                top_k_posterior=top_k_posterior,
                                                                                delta_likelihood=delta_likelihood)

        res_0 = vqvae.decode([x_0], start_level=2, bs_chunks=1).squeeze(-1)  # n_samples, sample_tokens*128
        res_1 = vqvae.decode([x_1], start_level=2, bs_chunks=1).squeeze(-1)  # n_samples, sample_tokens*128
        
        # remaining_0, remaining_1 = cross_prior_rejection(x_0, x_1, log_p_0_sum, log_p_1_sum,
        #                                                  [priors[0].prior, priors[1].prior],
        #                                                                       sample_tokens)
        
        end_sep = timer()
        sep_time = end_sep - start_sep
        
        start_rej = timer()

        remaining_0 = None
        remaining_1 = None
        # noinspection PyTypeChecker


        marginal_0, marginal_1, marginal_0_idx_sorted, marginal_1_idx_sorted, argmax_posterior_0, argmax_posterior_1 = rejection_sampling(
            log_p_0_sum, log_p_1_sum, res_0, res_1, remaining_0,
            remaining_1, mix, alpha, bs,
            rejection_sigma=rejection_sigma, n_samples=sample_tokens)
        
        end_rej = timer()
        rej_time = end_rej - start_rej
        # rejection_sampling_latent(log_p_0_sum, log_p_1_sum, log_likelihood_sum, bs)

        # (m0, m1) vs (res_0, res_1): comparison between source E->D and samples from the autoregressive priors
        sdr0, sdr1, sdr0_sorted, sdr1_sorted, sdr0_sorted_idx, sdr1_sorted_idx = evaluate_sdr(m0, m1, res_0, res_1)

        # (m0_real, m1_real) vs (res_0, res_1): comparison between source and samples from the autoregressive priors
        sdr0_real, sdr1_real, sdr0_real_sorted, sdr1_real_sorted, sdr0_real_sorted_idx, sdr1_real_sorted_idx = evaluate_sdr(m0_real, m1_real, res_0, res_1)

        marginal_idx0 = marginal_0_idx_sorted[-1]
        marginal_idx1 = marginal_1_idx_sorted[-1]

        if (sdr0[argmax_posterior_0].isnan() or
            sdr1[argmax_posterior_1].isnan() or
            sdr0_real[argmax_posterior_0].isnan() or
            sdr1_real[argmax_posterior_1].isnan() or

            sdr0[marginal_idx0].isnan() or
            sdr1[marginal_idx1].isnan() or
            sdr0_real[marginal_idx0].isnan() or
            sdr1_real[marginal_idx1].isnan() or

            sdr0_sorted[-1].isnan() or
            sdr1_sorted[-1].isnan() or
            sdr0_real_sorted[-1].isnan() or
            sdr1_real_sorted[-1].isnan()):
            print("Something is Nan...")
            continue

        ####################################SAVE AUDIO####################################

        torchaudio.save(str(chunk_path / f"original_1.wav"), orig1.view(-1).cpu(), sample_rate=22050)
        torchaudio.save(str(chunk_path / f"original_2.wav"), orig2.view(-1).cpu(), sample_rate=22050)
        torchaudio.save(str(chunk_path / f"mixture.wav"), mix_real.view(-1).cpu(), sample_rate=22050)

        torchaudio.save(str(chunk_path / f"res_0-{sdr0_real_sorted_idx[-1]}.wav"), res_0[sdr0_real_sorted_idx[-1]].view(-1).cpu(), sample_rate=22050)
        torchaudio.save(str(chunk_path / f"res_1-{sdr1_real_sorted_idx[-1]}.wav"), res_1[sdr1_real_sorted_idx[-1]].view(-1).cpu(), sample_rate=22050)
       
        ##################################################################################

        total_sdr0_gt_rejection_marginal += sdr0[marginal_idx0]  # sdr0[selected_marginal]
        total_sdr1_gt_rejection_marginal += sdr1[marginal_idx1]  # sdr1[selected_marginal]

        total_sdr0_real_rejection_marginal += sdr0_real[marginal_idx0]  # sdr0_real[selected_marginal]
        total_sdr1_real_rejection_marginal += sdr1_real[marginal_idx1]  # sdr1_real[selected_marginal]

        total_sdr0_gt_rejection_posterior += sdr0[argmax_posterior_0]  # sdr0[selected_marginal]
        total_sdr1_gt_rejection_posterior += sdr1[argmax_posterior_1]  # sdr1[selected_marginal]

        total_sdr0_real_rejection_posterior += sdr0_real[argmax_posterior_0]  # sdr0_real[selected_marginal]
        total_sdr1_real_rejection_posterior += sdr1_real[argmax_posterior_1]  # sdr1_real[selected_marginal]

        total_sdr0_gt += sdr0_sorted[-1]
        total_sdr1_gt += sdr1_sorted[-1]

        total_sdr0_real += sdr0_real_sorted[-1]
        total_sdr1_real += sdr1_real_sorted[-1]

        avg_time_sep += sep_time
        avg_time_rej += rej_time
        
        print(f'SDR GT: sdr0={sdr0_sorted[-1]}, sdr1={sdr1_sorted[-1]}')
        print(f'SDR REAL: sdr0={sdr0_real_sorted[-1]}, sdr1={sdr1_real_sorted[-1]}')

        # wandb.log({ "sdr0_gt_rejection_marginal": sdr0[marginal_idx0],
        #             "sdr1_gt_rejection_marginal": sdr1[marginal_idx1],
        #             "sdr0_real_rejection_marginal": sdr0_real[marginal_idx0],
        #             "sdr1_real_rejection_marginal": sdr1_real[marginal_idx1],

        #             "sdr0_gt_rejection_posterior": sdr0[argmax_posterior_0],
        #             "sdr1_gt_rejection_posterior": sdr1[argmax_posterior_1],
        #             "sdr0_real_rejection_posterior": sdr0_real[argmax_posterior_0],
        #             "sdr1_real_rejection_posterior": sdr1_real[argmax_posterior_1],
                    
        #             "sdr0_gt": sdr0_sorted[-1],
        #             "sdr1_gt": sdr1_sorted[-1],
        #             "sdr0_real": sdr0_real_sorted[-1],
        #             "sdr1_real": sdr1_real_sorted[-1],

        #             "sep_time": sep_time,
        #             "rej_time": rej_time,
        #             })        

    # wandb.log({ "sdr0_gt_rejection_marginal": total_sdr0_gt_rejection_marginal/len(dataset),
    #             "sdr1_gt_rejection_marginal": total_sdr1_gt_rejection_marginal/len(dataset),
    #             "sdr0_real_rejection_marginal": total_sdr0_real_rejection_marginal/len(dataset),
    #             "sdr1_real_rejection_marginal": total_sdr1_real_rejection_marginal/len(dataset),

    #             "sdr0_gt_rejection_posterior": total_sdr0_gt_rejection_posterior/len(dataset),
    #             "sdr1_gt_rejection_posterior": total_sdr1_gt_rejection_posterior/len(dataset),
    #             "sdr0_real_rejection_posterior": total_sdr0_real_rejection_posterior/len(dataset),
    #             "sdr1_real_rejection_posterior": total_sdr1_real_rejection_posterior/len(dataset),
                
    #             "sdr0_gt": total_sdr0_gt/len(dataset),
    #             "sdr1_gt": total_sdr1_gt/len(dataset),
    #             "sdr0_real": total_sdr0_real/len(dataset),
    #             "sdr1_real": total_sdr1_real/len(dataset),

    #             "avg_time_sep": avg_time_sep/len(dataset),
    #             "avg_time_rej": avg_time_rej/len(dataset),
    #             })
