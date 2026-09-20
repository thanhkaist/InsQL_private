# Instant Flow Q-Learning (InsQL)

Official implementation of:

**Compositional Average Velocity Modeling for Efficient Diffusion Offline Reinforcement Learning**
Thanh Nguyen, Abu Hanif Muhammad Syarubany, Hwanhee Kim, Chang D. Yoo — IROS 2026

## Overview

Diffusion policies are expressive but tie action generation to iterative denoising, which imposes heavy training cost and forces backpropagation through time (BPTT) across solver steps.

InsQL removes that bottleneck. Instead of learning the instantaneous *marginal* velocity of the flow, it parameterizes the policy with the **average velocity over finite intervals**, so the full noise-to-action map can be queried directly:

* ⚡ Single-step action generation during **both training and inference**
* 🧩 One policy network — no distillation, no multi-stage pipeline, no Jacobian computation
* 🔁 No BPTT in the actor update
* 🎯 Optional Q-guided candidate selection at test time

**Compositionality.** The average velocity over a long interval is the convex combination of average velocities over sub-intervals. InsQL trains on this recursion, anchored at the short-interval limit to the conditional velocity from flow matching — so the objective is fully self-contained and needs no pretrained teacher.

## 🛠️ Setup

```sh
conda create -n insql python=3.10 mesalib glew glfw pip=23 setuptools=63.2.0 wheel=0.38.4 protobuf=3.20 -c conda-forge -y
conda activate insql
```

### MuJoCo and `mujoco-py`
Follow the official guide [here](https://github.com/openai/mujoco-py#install-mujoco), or:

```sh
sudo apt-get update && sudo apt-get install -y wget tar libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf cmake
sudo ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so

USER_DIR=$HOME
wget -c "https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz"
mkdir -p $USER_DIR/.mujoco
cp mujoco210-linux-x86_64.tar.gz $USER_DIR/mujoco.tar.gz
rm mujoco210-linux-x86_64.tar.gz
tar -zxvf $USER_DIR/mujoco.tar.gz -C $USER_DIR/.mujoco

echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$USER_DIR/.mujoco/mujoco210/bin" >> ~/.bashrc
echo "export MUJOCO_PY_MUJOCO_PATH=$USER_DIR/.mujoco/mujoco210" >> ~/.bashrc
source ~/.bashrc
```

### Dependencies
```sh
pip install -r requirements.txt
pip install -e .
```

## 💻 Training

```sh
# MuJoCo locomotion
CUDA_VISIBLE_DEVICES=0 python pipelines/train_insql_d4rl_mujoco.py \
    task=halfcheetah-medium-expert-v2 mode=train seed=0 name=hc_me_s0 save_dir=results

# AntMaze navigation
CUDA_VISIBLE_DEVICES=0 python pipelines/train_insql_d4rl_antmaze.py \
    task=antmaze-medium-play-v2 mode=train seed=0 name=am_mp_s0 save_dir=results
```

Configs live in `configs/insql/<domain>/`, per-task settings in `configs/insql/<domain>/task/`. Checkpoints go to `<save_dir>/<pipeline_name>/<base_path>/<env_name>/`, where `base_path` encodes λ, η, sampling steps and seed so runs never collide.

Add `enable_wandb=true project=InsQL group=<group>` to log to Weights & Biases; logging is off by default.

## 📊 Evaluation

```sh
CUDA_VISIBLE_DEVICES=0 python pipelines/train_insql_d4rl_mujoco.py \
    task=halfcheetah-medium-v2 mode=inference seed=0 \
    name=hc_m_s0 save_dir=results ckpt=latest
```

Pass the same `task`, `seed` and `save_dir` used for training, since those determine the checkpoint path.

## ⚙️ Key Hyperparameters

| Config key | Paper symbol | Default | Meaning |
| --- | --- | --- | --- |
| `flow_ratio` | **λ** | 0.5 | Probability of using the boundary objective instead of the compositional one, i.e. the mixing weight in `L_BC = (1−λ)·L_cps + λ·L_bdr` |
| `num_candidates` | **K** | 5 | Q-guided candidate actions sampled per state at test time. Set `num_candidates=1` to disable Q-guided selection |
| `task.eta` | **η** | per task | Scale of the Q-term; the actor uses `α = η / E[‖Q‖]` |

Reproduce the λ ablation with `flow_ratio={1,0.75,0.5,0.25,0}` and the K ablation with `num_candidates={1,5,10,20,50}`.

## 📌 Benchmark Tasks

| Domain | Datasets |
| --- | --- |
| **MuJoCo** | halfcheetah / hopper / walker2d × medium, medium-replay, medium-expert |
| **AntMaze** | antmaze-{medium,large}-{play,diverse}-v2 |

Per-task η and `weight_temperature` are set in each task YAML.

**Not included in this repository:** the multimodal checkerboard toy study and the real-world robotic manipulation experiments (Sawyer arm; Pickup Banana and Sweep Cube).

## Citation

```bibtex
@inproceedings{
    nguyen2026compositional,
    title={Compositional Average Velocity Modeling for Efficient Diffusion Offline Reinforcement Learning},
    author={Nguyen, Thanh and Syarubany, Abu Hanif Muhammad and Kim, Hwanhee and Yoo, Chang D},
    booktitle={IROS},
    year={2026},
}
```

## 🏷️ Acknowledgements
Built upon [CleanDiffuser](https://github.com/CleanDiffuserTeam/CleanDiffuser) and [Habi](https://github.com/Josh00-Lu/Habi.git); please cite these works as well. See the [license](LICENSE).
