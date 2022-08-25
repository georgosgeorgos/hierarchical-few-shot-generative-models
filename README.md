## Hierarchical Context Aggregation for Few-Shot Generation

*A few-shot generative model should be able to generate data from a novel distribution by only observing a limited set of examples. In few-shot learning the model is trained on data from many sets from distributions sharing some underlying properties such as sets of characters from different alphabets or objects from different categories. We extend current latent variable models for sets to a fully hierarchical approach with an attention-based point to set-level aggregation and call our method SCHA-VAE for Set-Context-Hierarchical-Aggregation Variational Autoencoder. We explore likelihood-based model comparison, iterative data sampling, and adaptation-free out-of-distribution generalization. Our results show that the hierarchical formulation better captures the intrinsic variability within the sets in the small data regime. This work generalizes deep latent variable approaches to few-shot learning, taking a step toward large-scale few-shot generation with a formulation that readily works with current state-of-the-art deep generative models*.

This repo contains code and experiments for: 

> **SCHA-VAE: Hierarchical Context Aggregation for Few-Shot Generation** \
> [Giorgio Giannone](https://georgosgeorgos.github.io/), [Ole Winther](https://olewinther.github.io/) \
> ICML 2022

* paper: https://proceedings.mlr.press/v162/giannone22a.html
* [page](https://georgosgeorgos.github.io/hierarchical-few-shot-generative-models/)

and

> **Hierarchical Few-Shot Generative Models** \
> Giorgio Giannone, Ole Winther \
> MetaLearn21

* paper: https://openreview.net/forum?id=INSai0E0VXN
* [page](https://georgosgeorgos.github.io/hierarchical-few-shot-generative-models/)

-------
## Settings

Clone the repo:
```bash
git clone https://github.com/georgosgeorgos/hierarchical-few-shot-generative-models
cd hierarchical-few-shot-generative-models
```

Create and activate the conda env:
```bash
conda env create -f environment.yml
conda activate hfsgm
```

The code has been tested on Ubuntu 18.04, Python 3.6 and CUDA 11.3

We use `wandb` for visualization. 
The first time you run the code you will need to login.

## Data

We provide preprocessed Omniglot dataset in `data`.
If you want to try CelebA you first need to download the dataset from [here](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg).


## Dataset
In `dataset` we provide utilities to process and augment datasets in the few-shot setting. 
Each dataset is a large collection of small sets. Sets can be created dynamically.
The `dataset/base.py` file collects basic info about the datasets.
For binary datasets (`omniglot_ns.py`) we augment using flipping and rotations. For RGB datasets (`celeba.py`) we use only flipping.

## Experiment

In `experiment` we implement scripts for model evaluation, experiments and visualizations.

* `attention.py` - visualize attention weights and heads for models with learnable aggregations (LAG).
* `cardinality.py` - compute ELBOs for different input set size: [1, 2, 5, 10, 20].
* `classifier_mnist.py` - few-shot classifiers on MNIST.
* `kl_layer.py` - compute KL over z and c for each layer in latent space. 
* `marginal.py` - compute approximate log-marginal likelihood with 1K importance samples.
* `refine_vis.py` - visualize refined samples.
* `sampling_rgb.py` - reconstruction, conditional, refined, unconditional sampling for RGB datasets.
* `sampling_transfer.py` - reconstruction, conditional, refined, unconditional sampling on transfer datasets.
* `sampling.py` - reconstruction, conditional, refined, unconditional sampling for binary datasets.
* `transfer.py` - compute ELBOs on MNIST, DoubleMNIST, TripleMNIST.

## Model
In `model` we implement baselines and model variants.

* `base.py` - base class for all the models.
* `vae.py` - Variational Autoencoder (VAE).
* `ns.py` - Neural Statistician (NS).
* `tns.py` - NS with learnable aggregation (NS-LAG).
* `cns.py` - NS with convolutional latent space (CNS).
* `ctns.py` - CNS with learnable aggregation (CNS-LAG).
* `hfsgm.py` - Hierarchical Few-Shot Generative Model (HFSGM).
* `thfsgm.py` - HFSGM with learnable aggregation (HFSGM-LAG).
* `chfsgm.py` - HFSGM with convolutional latent space (CHFSGM).
* `cthfsgm.py` - CHFSGM with learnable aggregation (CHFSGM-LAG).
* `chfsgm_multi.py` - Set-Context-Hierarchical-Aggregation Variational Autoencoder (SCHA-VAE).

## Script
Scripts used for training the models in the paper.

To run a SCHA-VAE on Omniglot:

```bash
sh script/main_chfsgm_multi.sh GPU_NUMBER omniglot_ns
```

------
## Train a model

To train a generic model run:

```python
python main.py --name {VAE, NS, CNS, CTNS, CHFSGM, CTHFSGM, CHFSGM_MULTISCALE} \
               --model {vae, ns, cns, ctns, chfsgm, cthfsgm, chfsgm_multiscale} \
               --augment \
               --dataset omniglot_ns \
               --likelihood binary \
               --hidden-dim 128 \
               --c-dim 32 \
               --z-dim 32 \
               --output-dir /output \
               --alpha-step 0.98 \
               --alpha 2 \
               --adjust-lr \
               --scheduler plateau \
               --sample-size {2, 5, 10} \
               --sample-size-test {2, 5, 10} \
               --num-classes 1 \
               --learning-rate 1e-4 \
               --epochs 400 \
               --batch-size 100 \
               --tag (optional string)
```

If you do not want to save logs, use the flag `--dry_run`. This flag will call `utils/trainer_dry.py` instead of `trainer.py`.

--------
## Acknowledgments

A lot of code and ideas borrowed from:

* https://github.com/conormdurkan/neural-statistician
* https://github.com/addtt/ladder-vae-pytorch
* https://github.com/vlievin/biva-pytorch
* https://github.com/didriknielsen/survae_flows
* https://github.com/openai/vdvae


## Citations

```bibtex
@inproceedings{Giannone2022SCHAVAEHC,
  title={SCHA-VAE: Hierarchical Context Aggregation for Few-Shot Generation},
  author={Giorgio Giannone and Ole Winther},
  booktitle={ICML},
  year={2022}
}
```

```bibtex
@article{Giannone2021HierarchicalFG,
  title={Hierarchical Few-Shot Generative Models},
  author={Giorgio Giannone and Ole Winther},
  journal={ArXiv},
  year={2021},
  volume={abs/2110.12279}
}
```
