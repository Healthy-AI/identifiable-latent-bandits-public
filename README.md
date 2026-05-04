# Identifiable Latent Bandits
Code for the paper [Identifiable Latent Bandits](https://arxiv.org/abs/2407.16239).

This repository contains the code used to train identifiable latent variable
models; run the context prior greedy (CPG), full prior greedy (FPG), and
baseline bandit algorithms; and reproduce the synthetic and semi-synthetic ADCB
experiments from the paper. The code builds on the
[ICE-BeeM](https://github.com/ilkhem/icebeem) repository.

## Repository Layout

```text
bandits/                 Bandit environments, algorithms, and inference wrappers
bandits/ADCB/            ADCB simulator assets and helpers
configs/                 Experiment configs for LVM and bandit runs
data/                    Synthetic ILB generator and ADCB preprocessing
metrics/                 Mean correlation coefficient (MCC) metric
models/                  ILB, VAE, regression, and MLP model implementations
runners/                 Training and evaluation functions called by lvm_run.py
bandits_run.py           Runs bandit experiments from a trained LVM checkpoint
bandits_eval.py          Plotting and aggregation helpers for bandit results
lvm_run.py               Trains or evaluates latent variable models
```

## Installation

The pinned Python dependencies are listed in `requirements.txt`. A clean virtual
environment is recommended.

The current requirements use `TensorFlow 2.15.1` and `PyTorch 2.1.2`. 
If you are rerunning older checkpoints, keep the TensorFlow/PyTorch stack
aligned with the environment that produced those checkpoints.

## Running Latent Variable Model Experiments

Use `lvm_run.py` for training and evaluating latent variable models. The
`--config` argument is resolved relative to `configs/`.

```bash
python lvm_run.py \
  --dataset ILB \
  --method ILB \
  --config lvm-ilb.yaml \
  --nSims 10 \
  --run experiments/ILB-example \
  --start_seed 0
```

Available datasets are:

- `ILB`: synthetic identifiable latent bandit data from `data/ilb.py`
- `ADCB`: CSV-backed ADCB-style data from `datasets/`
- `nonlinear`: nonlinear synthetic ablations in the paper

Available methods are:

- `ILB`: Our contastive learning algorithm
- `VAE`: variational autoencoder baseline
- `regression`: sequence/regression baseline
- `mlp`: MLP baseline

Outputs are written under:

```text
<run>/checkpoints/<method>/<dataset>/<n_layers>/<n_obs_per_seg>/<seed>/
```

For example:

```text
experiments/ILB-example/checkpoints/ILB/ILB/2/200/0/
```

The runner also stores a copy of the config at:

```text
<run>/checkpoints/<method>/config.yaml
```

This copy is important because `bandits_run.py` reads it later from the LVM
checkpoint path.

To evaluate an existing LVM run, set `EVAL: true` in the corresponding config
and point `--run` at the same run directory:

```bash
python lvm_run.py \
  --dataset ILB \
  --method ILB \
  --config lvm-ilb.yaml \
  --nSims 10 \
  --run _experiments/ILB-example \
  --start_seed 0
```

## Running Bandit Experiments

Bandit experiments are configured through `configs/bandit-*.yaml`. Unlike
`lvm_run.py`, `bandits_run.py` expects the full config path.

```bash
python bandits_run.py \
  --config configs/bandit-ilb.yaml \
  --start_seed 0 \
  --end_seed 10
```

The bandit config controls:

- `lvm_path`: checkpoint directory produced by `lvm_run.py`
- `lvm`: inference backend, one of `lvm`, `vae`, or `oracle`
- `dftype`: data type, usually `ilb` or `adcb`
- `n_iter`: number of bandit interaction rounds
- `return_ica`: whether to use the linear ICA post-processing for the latent representation
- `bandits`: which algorithms to run
- `bandit_config`: algorithm-specific hyperparameters

Implemented bandit algorithms include `mab`, context prior greedy (CPG) `greedy1` , full prior greedy (FPG) `greedy2`,
`projected_thompson`, `regression`, `mab_prior`, `random_feature_bandit`,
and `linear_thompson`.

Each result is saved as a pickle under:

```text
<lvm_path>/bandit_results/<bandit>/seed_<seed>*.bin
```

Existing seed files are skipped, which makes interrupted runs easy to resume.

## Plotting Results

`bandits_eval.py` contains helpers for aggregating regret files and producing
the paper plots. The most useful functions are:

- `save_regret_pickle(banditdir, bandit)`: stacks per-seed `.bin` files into a
  regret array
- `cum_regret_plot(...)`: plots cumulative regret
- `synthetic_plot(...)`: reproduces the synthetic bandit plot style

The bottom of `bandits_eval.py` contains paper-specific path blocks and
executable plotting calls. Update or comment those blocks before running or
importing the file, then call the desired helpers after the relevant
`bandit_results/` directories exist.

## Citation
```bib
@misc{balcioglu2026ILB,
      title={Identifiable Latent Bandits: Leveraging observational data for personalized decision-making}, 
      author={Ahmet Zahid Balcıoğlu and Newton Mwai and Emil Carlsson and Fredrik D. Johansson},
      year={2025},
      eprint={2407.16239},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.16239},
}
```

## Notes

- Synthetic ILB data is generated from the LVM configs at run time.
- ADCB-style runs use CSV files in `datasets/` and column definitions in
  `configs/data_configs/adcb_config.yaml`.
- For bandit runs, the LVM checkpoint path should have the standard structure
  produced by `lvm_run.py`, since the bandit runner locates the saved LVM config
  relative to that path.

## License

This repository is released under the MIT License, see [LICENSE](LICENSE). Portions derived from ICE-BeeM retain their original MIT
notice:

```
MIT License

Copyright (c) 2020 Ilyes Khemakhem


Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```