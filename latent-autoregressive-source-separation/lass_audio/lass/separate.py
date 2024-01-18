import abc
from pathlib import Path
from typing import Callable, Sequence, Mapping
from time import time
import diba
import numpy as np
import torch
import torchaudio
# import wandb
from diba.diba import Likelihood
from torch.utils.data import DataLoader
from diba.interfaces import SeparationPrior

from lass.datasets import SeparationDataset
from lass.datasets import SeparationSubset
from lass.diba_interfaces import JukeboxPrior, SparseLikelihood
from lass.utils import assert_is_audio, decode_latent_codes, get_dataset_subsample, get_raw_to_tokens, setup_priors, setup_vqvae, evaluate_sdr
from lass.datasets import ChunkedPairsDataset
from jukebox.utils.dist_utils import setup_dist_from_mpi


audio_root = Path(__file__).parent.parent


class Separator(torch.nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()
        
    @abc.abstractmethod
    def separate(mixture) -> Mapping[str, torch.Tensor]:
        ...


class BeamsearchSeparator(Separator):
    def __init__(
        self,
        encode_fn: Callable,
        decode_fn: Callable,
        priors: Sequence[SeparationPrior], 
        likelihood: Likelihood, 
        num_beams: int,
    ):
        super().__init__()
        self.likelihood = likelihood
        self.source_types = list(priors)
        self.priors = list(priors)
        self.num_beams = num_beams

        self.encode_fn = encode_fn #lambda x: vqvae.encode(x.unsqueeze(-1), vqvae_level, vqvae_level + 1).view(-1).tolist()
        self.decode_fn = decode_fn #lambda x: decode_latent_codes(vqvae, x.squeeze(0), level=vqvae_level)

    @torch.no_grad()
    def separate(self, mixture: torch.Tensor) -> Mapping[str, torch.Tensor]:
        # convert signal to codes
        mixture_codes = self.encode_fn(mixture) 

        # separate mixture (x has shape [2, num. tokens])
        x0, x1 = diba.fast_beamsearch_separation(
            priors=self.priors,
            likelihood=self.likelihood,
            mixture=mixture_codes,
            num_beams=self.num_beams,
        )

        print(f'Shape di x1: {x0.shape}')
        print(f'Shape di x1: {x1.shape}')
        print(f"Shape di decoded: {self.decode_fn(x0[-1:]).shape}")
        decode_x0 = torch.zeros((x0.shape[0], 65536))
        decode_x1 = torch.zeros((x0.shape[0], 65536))

        for i, sample in enumerate(x0):
            decode_x0[i] = self.decode_fn(x0[i])
            decode_x1[i] = self.decode_fn(x1[i])

        # decode results
        return  decode_x0, decode_x1
        # return self.decode_fn(x0[-1:]), self.decode_fn(x1[-1:])
    
class TopkSeparator(Separator):
    def __init__(
        self,
        encode_fn: Callable,
        decode_fn: Callable,
        priors: Sequence[SeparationPrior],
        likelihood: Likelihood, 
        num_samples: int,
        temperature: float = 1.0,
        top_k: int = None,
        sample_tokens: int = 1024,
    ):
        super().__init__()
        self.likelihood = likelihood
        self.source_types = list(priors)
        self.priors = priors
        self.num_samples = num_samples
        self.temperature = temperature
        self.top_k = top_k
        self.sample_tokens = sample_tokens

        self.encode_fn = encode_fn
        self.decode_fn = decode_fn

    def separate(self, mixture: torch.Tensor):
        mixture_codes = self.encode_fn(mixture) 

        x_0, x_1 = diba.fast_sampled_separation(
            priors=self.priors,
            likelihood=self.likelihood,
            mixture=mixture_codes,
            num_samples=self.num_samples,
            temperature=self.temperature,
            top_k=self.top_k,
        )

        res_0 = torch.zeros((x_0.shape[0], self.num_samples*self.sample_tokens))
        res_1 = torch.zeros((x_0.shape[0], self.num_samples*self.sample_tokens))

        for i, sample in enumerate(x_0):
            res_0[i] = self.decode_fn(x_0[i])
            res_1[i] = self.decode_fn(x_1[i])

        return res_0, res_1

# -----------------------------------------------------------------------------


@torch.no_grad()
def separate_dataset(
    dataset: SeparationDataset,
    separator: Separator,
    save_path: Path,
    resume: bool = False,
    num_workers: int = 0,
):

    # get chunks
    loader = DataLoader(dataset, batch_size=1, num_workers=num_workers)

    # main loop
    save_path.mkdir(exist_ok=True)
    
    stop_at = 0
    sep_times = []

    for batch_idx, batch in enumerate(loader):
        
        chunk_path = save_path / f"{batch_idx}"
        if chunk_path.exists():
            print(f"Skipping path: {chunk_path}")
            continue

        # load audio tracks
        origs = batch
        ori_0, ori_1 = origs
        print(f"chunk {batch_idx+1} out of {len(dataset)}")

        start_sep = time()
        # generate mixture
        mixture = 0.5 * ori_0 + 0.5 * ori_1
        mixture = mixture.squeeze(0) # shape: [1 , sample-length]
        res0, res1 = separator.separate(mixture=mixture)

        end_sep = time()
        sep_time = end_sep - start_sep 

        chunk_path.mkdir(parents=True)

        # evaluate generated results
        sdr_0, sdr_1, sdr_0_sorted, sdr_1_sorted, sdr_0_sorted_idx, sdr_1_sorted_idx = evaluate_sdr(ori_0.squeeze(0), ori_1.squeeze(0), res0, res1)

        print(f"SDR track {batch_idx}:\nsdr_0:{sdr_0[-1]}\nsdr_1:{sdr_1[-1]}\nsdr_0_max:{torch.max(sdr_0)}\nsdr_1_max:{torch.max(sdr_1)}\nsep_time: {end_sep - start_sep}")
        
        if (sdr_0[-1].isnan() or sdr_1[-1].isnan() or
            sdr_0_sorted[-1].isnan() or sdr_1_sorted[-1].isnan()):
                
                print("Something is Nan...")
                continue
        
        torchaudio.save(str(chunk_path / f"original_1.wav"), ori_0.view(-1).cpu(), sample_rate=22050)
        torchaudio.save(str(chunk_path / f"original_2.wav"), ori_1.view(-1).cpu(), sample_rate=22050)
        torchaudio.save(str(chunk_path / f"mixture.wav"), mixture.view(-1).cpu(), sample_rate=22050)

        torchaudio.save(str(chunk_path / f"res_0-{sdr_0_sorted_idx[-1]}.wav"), res0[sdr_0_sorted_idx[-1]].view(-1).cpu(), sample_rate=22050)
        torchaudio.save(str(chunk_path / f"res_1-{sdr_1_sorted_idx[-1]}.wav"), res1[sdr_1_sorted_idx[-1]].view(-1).cpu(), sample_rate=22050)
    
        # wandb.log({'sdr_0_real':sdr_0_sorted[-1],
        #            'sdr_1_real':sdr_1_sorted[-1],
        #            'sep_time':sep_time})
        
        del res0, res1, origs
    
# -----------------------------------------------------------------------------

def main(
    audio_dir_1: str = "../../test/bass",
    audio_dir_2: str = "../../test/drums",
    vqvae_path: str = "./logs/vq_vae/checkpoint_step_19160.pth.tar",
    prior_1_path: str = "./logs/lass_prior_bass/checkpoint_latest.pth.tar",
    prior_2_path: str = "./logs/lass_prior_drums/checkpoint_latest.pth.tar",
    
    sum_frequencies_path: str = "./logs/vqvae_sum_distribution/sum_dist_10000.npz",

    vqvae_type: str = "vqvae",
    prior_1_type: str = "small_prior",
    prior_2_type: str = "small_prior",
    max_sample_tokens: int = 1024,
    sample_rate: int = 22050,
    save_path: str = "lass/results-test",
    resume: bool = False,
    **kwargs,
):
    # convert paths
    save_path = Path(save_path)
    audio_dir_1 = Path(audio_dir_1)
    audio_dir_2 = Path(audio_dir_2)

    rank, local_rank, device = setup_dist_from_mpi(port=29533, verbose=True)

    # setup models
    vqvae = setup_vqvae(
        vqvae_path=vqvae_path,
        vqvae_type=vqvae_type,
        sample_rate=sample_rate,
        sample_tokens=max_sample_tokens,
        device=device,
    )

    priors = setup_priors(
        prior_paths=[prior_1_path, prior_2_path],
        prior_types=[prior_1_type, prior_2_type],
        vqvae=vqvae,
        fp16=True,
        device=device,
    )

    # create separator
    level = vqvae.levels - 1
   
    separator = TopkSeparator(
        encode_fn=lambda x: vqvae.encode(x.unsqueeze(-1).to(device), level, level + 1)[-1].squeeze(0).tolist(), 
        decode_fn=lambda x: decode_latent_codes(vqvae, x.squeeze(0), level=level),
        priors=[JukeboxPrior(p.prior, torch.zeros((), dtype=torch.float32, device=device)) for p in priors],
        likelihood=SparseLikelihood(sum_frequencies_path, device, 3.0),
        num_samples=64,
        top_k=None,
        sample_tokens=max_sample_tokens)


    # setup dataset
    raw_to_tokens = get_raw_to_tokens(vqvae.strides_t, vqvae.downs_t)
    dataset = ChunkedPairsDataset(
        instrument_1_audio_dir=audio_dir_1,
        instrument_2_audio_dir=audio_dir_2,
        sample_rate=sample_rate,
        max_chunk_size=raw_to_tokens * max_sample_tokens,
        min_chunk_size=raw_to_tokens,
    )


    # wandb.init(
    #     id = "lass_evaluation", 
    #     # resume="must",
    #     # set the wandb project where this run will be logged
    #     project="DL-Project",
    #     name="lass_evaluation",
    #     # track hyperparameters and run metadata
    #     config={
    #     "dataset": "Slakh2100-Drums&Bass",
    #     "test-size": 90,
    #     "sr": 22050,
    #     "levels": 3,
    #     "l_bins": 2048,
    #     "raw_to_tokens":raw_to_tokens,
    #     "sample_tokens":max_sample_tokens,
    #     }
    # )

    # separate dataset
    separate_dataset(
        dataset=dataset,
        separator=separator,
        save_path=save_path,
        resume=resume,
    )

if __name__ == "__main__":
    main()