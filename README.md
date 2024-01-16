# LQVAE-LASS-hybrid
Project for the exam of Deep Learning &amp; Applied AI @Sapienza [2022/2023]

**Task**: Students will be required to compare the source separation music performance of the **LQ-VAE** model and the **LASS** model by re-training both models using publicly available datasets. They will then train a model with a loss function as in LQ-VAE, but using the technique of counting occurrences in the model codebook at inference time, as is done in LASS. The project aims to assess whether this hybrid approach can lead to better separation performance while maintaining efficiency at inference time.

This repository contains the following three models:

*   [LQVAE](https://github.com/michelemancusi/LQVAE-separation) | [paper](https://arxiv.org/abs/2110.05313)
*   [LASS](https://github.com/gladia-research-group/latent-autoregressive-source-separation) | [paper](https://arxiv.org/abs/2301.08562)
*   Hybrid

All models leverage their architecture from the paper [Jukebox: A Generative Model for Music](https://arxiv.org/abs/2005.00341).

## Implementation

The project was completed using **Google Colab**'s hardware. The GPU at my disposal could not have achieved the same results in sensible time compared to the T4 offered by the service. However the time limit of 3 hours per day limited the amount of training and testing available, even with several restarts, that's why the results obtained do not mirror the original work. 

## Data

The data used to train the models is from [Synthesized Lakh (Slakh) Dataset](http://www.slakh.com/), specifically focusing on bass and drums instruments. From the entire dataset only 600 songs, each sampled at 22050Hz, were selected from the complete dataset, 300 for bass and another 300 for drums. This individual sources were then paired to form a mixture, resulting in 300 mixtures. This straightforward process is detailed in the code *'slakh_scrape.py'*. Finally the mixtures and corresponding sources (bass and drums) were finally divided into 210 samples for training and 90 samples for testing. The reduction in the number of samples is attributed to limitations in computational resources.

