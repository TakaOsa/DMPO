# Offline Reinforcement Learning with Mixture of Deterministic Policies

PyTorch implementation of Deterministic mixture policy optimization (DMPO). If you use our code or data please cite the [paper](https://openreview.net/pdf?id=zkRCp4RmAF).

Method is tested on tasks in [D4RL](https://github.com/Farama-Foundation/D4RL). 
Networks are trained using [PyTorch 1.13](https://github.com/pytorch/pytorch) and Python 3.7. 

### Usage
For an experiment with DMPO *without* the mutual information regularization on the walker2d-expert-v2 task, run
```
python main_training.py --policy DMPO --env walker2d-expert-v2  --scale 10.0
```
We use the `--scale 10` for mujoco-v2 tasks.

For an experiment with DMPO *with* the mutual information regularization, use `--info-reg` option:
```
python main_training.py --policy DMPO --env walker2d-expert-v2 --scale 10.0 --info-reg
```
In DMPO, the discrete latent variable is sampled using Gumbel-softmax trick (see [Jiang et al. 2017](https://openreview.net/forum?id=rkE3y85ee) or [Maddison et al., 2017](https://openreview.net/forum?id=S1jE5L5gl)  ), and our implementation is based on [this repository](https://github.com/Schlumberger/joint-vae/).

While the implementation of DMPO in `DMPO.py` is simple, we incorporated some techniques in `DMPO_ant.py` to improve the performance for antmaze tasks.
Please refer to the appendix of the paper for details of the version for antmaze tasks.
For an experiment on the antmaze-large-play-v0 task, run
```
python main_training.py --policy DMPO_ant --env antmaze-large-play-v0 --scale 5.0 --info-reg
```
We use the `--scale 5.0` for antmaze tasks.

Our codes include our implementaion of baseline methods which we used in our [paper](https://openreview.net/pdf?id=zkRCp4RmAF), such as AWAC, IQL, mixAWAC, and LP-AWAC.
For example, if you would like to reproduce the result of mixAWAC, please run
```
python main_training.py --policy mixAWAC --env walker2d-expert-v2 --scale 10
```
In our implementation of AWAC, we used the double-clipped Q-learning, state-normalization and advantage normalization.

mixAWAC is a variant of AWAC, where the policy is given by a mixture of Gaussian policies.
In mixAWAC, the discrete latent variable is sampled using Gumbel-softmax trick as in DMPO.

LP-AWAC employs the latent-actor proposed in [LAPO](https://github.com/pcchenxi/LAPO-offlienRL).

### Bibtex
When you use our codes for your work, please cite:

```
@article{osa2023dmpo,
  title={Offline Reinforcement Learning with Mixture of Deterministic Policies},
  author={Takayuki Osa and Akinobu Hayashi and Pranav Deo and Naoki Morihira and Takahide Yoshiike},
  journal={Transactions on Machine Learning Research},
  year={2023}
}
```