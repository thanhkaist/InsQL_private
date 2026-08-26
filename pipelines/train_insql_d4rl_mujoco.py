import os
from copy import deepcopy

import d4rl
import gym
import hydra, wandb, uuid
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion import InsQL
from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_diffusion import InsQLMlp
from cleandiffuser.utils import report_parameters, DQLCritic, FreezeModules
from utils import set_seed
from tqdm import tqdm
from omegaconf import OmegaConf
import time


@hydra.main(config_path="../configs/insql/mujoco", config_name="mujoco", version_base=None)
def pipeline(args):
    args.device = args.device if torch.cuda.is_available() else "cpu"
    if args.enable_wandb and args.mode in ["inference", "train"]:
        wandb.require("core")
        print(args)
        wandb.init(
            reinit=True,
            id=str(uuid.uuid4()),
            project=str(args.project),
            group=str(args.group),
            name=str(args.name),
            config=OmegaConf.to_container(args, resolve=True)
        )

    set_seed(args.seed)
    base_path = "insql_d4rl_mujoco"
    base_path += f"_td{args.time_dist}"
    base_path += f"_eta{args.task.eta}"
    base_path += f"_ss{args.train_sampling_steps}"
    base_path += f"_fr{args.flow_ratio}"
    base_path += f"_ema{int(args.use_ema)}"
    base_path += f"_wu{args.warm_up_steps}"
    base_path += f"_ad{args.adaptive_l2_loss}"
    base_path += f"_seed{args.seed}"


    
    save_path = f'{args.save_dir}/{args.pipeline_name}/'+ base_path+ f'/{args.task.env_name}/'
    video_path = f'video_outputs/{args.pipeline_name}/'+ base_path+ f'/{args.task.env_name}/'
    os.makedirs(save_path, exist_ok=True)
    if args.mode == "visual":
        os.makedirs(video_path, exist_ok=True)

    # ---------------------- Create Dataset ----------------------
    env = gym.make(args.task.env_name)
    dataset = D4RLMuJoCoTDDataset(d4rl.qlearning_dataset(env), args.normalize_reward)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

    # --------------- Network Architecture -----------------
    nn_diffusion = InsQLMlp(obs_dim, act_dim, emb_dim=64, timestep_emb_type="positional").to(args.device)
    nn_condition = IdentityCondition(dropout=0.0).to(args.device)

    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"==============================================================================")

    # --------------- Diffusion Model Actor --------------------
    actor = InsQL(
        nn_diffusion, nn_condition, optim_params={"lr": args.actor_learning_rate},
        x_max=+1. * torch.ones((1, act_dim), device=args.device),
        x_min=-1. * torch.ones((1, act_dim), device=args.device), ema_rate=args.ema_rate, device=args.device, time_dist=[args.time_dist,-0.4,1],adaptive_l2_loss=args.adaptive_l2_loss,flow_ratio=args.flow_ratio, warm_up_steps=args.warm_up_steps)

    # ------------------ Critic ---------------------
    critic = DQLCritic(obs_dim, act_dim, hidden_dim=args.hidden_dim).to(args.device)
    critic_target = deepcopy(critic).requires_grad_(False).eval()
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_learning_rate)

    # ---------------------- Training ----------------------
    if args.mode == "train":

        actor_lr_scheduler = CosineAnnealingLR(actor.optimizer, T_max=args.gradient_steps)
        critic_lr_scheduler = CosineAnnealingLR(critic_optim, T_max=args.gradient_steps)

        actor.train()
        critic.train()

        n_gradient_step = 0
        log = {"bc_loss": 0., "q_loss": 0., "critic_loss": 0., "target_q_mean": 0., "fm_loss": 0.}

        # == For eval ==
        env_eval = gym.vector.make(args.task.env_name, args.num_envs)
        normalizer_eval = dataset.get_normalizer()
        prior_eval = torch.zeros((args.num_envs * args.num_candidates, act_dim), device=args.device)
        # == end for eval ==
        
        prior = torch.zeros((args.batch_size, act_dim), device=args.device)
        pbar = tqdm(total=args.gradient_steps / args.log_interval)
        for batch in loop_dataloader(dataloader):

            obs, next_obs = batch["obs"]["state"].to(args.device), batch["next_obs"]["state"].to(args.device)
            act = batch["act"].to(args.device)
            rew = batch["rew"].to(args.device)
            tml = batch["tml"].to(args.device)

            # Critic Training
            current_q1, current_q2 = critic(obs, act)

            next_act, _ = actor.sample(
                prior, solver=args.solver,
                n_samples=args.batch_size, sample_steps=args.train_sampling_steps, use_ema=args.use_ema,
                temperature=1.0, condition_cfg=next_obs, w_cfg=1.0, requires_grad=False)

            target_q = torch.min(*critic_target(next_obs, next_act))
            target_q = (rew + (1 - tml) * args.discount * target_q).detach()

            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()

            # -- Policy Training
            bc_loss, is_flow_loss = actor.loss(act,condition = obs)
            new_act, _ = actor.sample(
                prior, solver=args.solver,
                n_samples=args.batch_size, sample_steps=args.train_sampling_steps, use_ema=False,
                temperature=1.0, condition_cfg=obs, w_cfg=1.0, requires_grad=True)

            with FreezeModules([critic, ]):
                q1_new_action, q2_new_action = critic(obs, new_act)
            if np.random.uniform() > 0.5:
                q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
            else:
                q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
            actor_loss = bc_loss + args.task.eta * q_loss

            actor.optimizer.zero_grad()
            actor_loss.backward()
            actor.optimizer.step()

            actor_lr_scheduler.step()
            critic_lr_scheduler.step()

            # -- ema
            if n_gradient_step % args.ema_update_interval == 0:
                if n_gradient_step >= 1000:
                    actor.ema_update()
                for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                    target_param.data.copy_(0.995 * param.data + (1 - 0.995) * target_param.data)

            # # ----------- Logging ------------
            if is_flow_loss:
                log["fm_loss"] += bc_loss.item()
            else:
                log["bc_loss"] += bc_loss.item()
            
            log["q_loss"] += q_loss.item()
            log["critic_loss"] += critic_loss.item()
            log["target_q_mean"] += target_q.mean().item()

            if (n_gradient_step + 1) % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["bc_loss"] /= args.log_interval
                log["fm_loss"] /= args.log_interval
                log["q_loss"] /= args.log_interval
                log["critic_loss"] /= args.log_interval
                log["target_q_mean"] /= args.log_interval
                print(log)
                if args.enable_wandb:
                    wandb.log(log, step=n_gradient_step + 1)
                pbar.update(1)
                log = {"bc_loss": 0., "q_loss": 0., "critic_loss": 0., "target_q_mean": 0., "fm_loss": 0.}

            if (n_gradient_step + 1) % args.eval_interval == 0:
                # Online evaluation 
                actor.eval()
                critic.eval()
                critic_target.eval()

                
                episode_rewards = []

                t0 = time.time()
                for i in range(1):

                    obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0

                    while not np.all(cum_done) and t < 1000 + 1:
                        # normalize obs
                        obs = torch.tensor(normalizer_eval.normalize(obs), device=args.device, dtype=torch.float32)
                        obs = obs.unsqueeze(1).repeat(1, args.num_candidates, 1).view(-1, obs_dim)

                        # sample actions
                        with torch.no_grad():
                            act, _ = actor.sample(
                                prior_eval,
                                solver=args.solver,
                                n_samples=args.num_envs * args.num_candidates,
                                sample_steps=1,
                                condition_cfg=obs, w_cfg=1.0,
                                use_ema=args.use_ema, temperature=args.temperature)

                        # resample
                        with torch.no_grad():
                            q = critic_target.q_min(obs, act)
                            q = q.view(-1, args.num_candidates, 1)
                            w = torch.softmax(q * args.task.weight_temperature, 1)
                            act = act.view(-1, args.num_candidates, act_dim)

                            indices = torch.multinomial(w.squeeze(-1), 1).squeeze(-1)
                            sampled_act = act[torch.arange(act.shape[0]), indices].cpu().numpy()

                        # step
                        obs, rew, done, info = env_eval.step(sampled_act)

                        t += 1
                        cum_done = done if cum_done is None else np.logical_or(cum_done, done)
                        ep_reward += (rew * (1 - cum_done)) if t < 1000 else rew
                        # print(f'[t={t}] rew: {np.around((rew * (1 - cum_done)), 2)}')

                        if np.all(cum_done):
                            break

                    episode_rewards.append(ep_reward)

                episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
                episode_rewards = np.array(episode_rewards).reshape(-1) * 100
                mean = np.mean(episode_rewards)
                err = np.std(episode_rewards) / np.sqrt(len(episode_rewards))
                t1 = time.time() - t0
                tqdm.write(f"Mean: {mean:.3f}, Error: {err:.3f}, Eval duration: {t1:.2f} seconds")

                actor.train()
                critic.train()
                critic_target.train()

                if args.enable_wandb:
                    wandb.log({'Mean Reward 1 step': mean, 'Error 1 step': err})


            # ----------- Saving ------------
            if (n_gradient_step + 1) % args.save_interval == 0:
                actor.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
                actor.save(save_path + f"diffusion_ckpt_latest.pt")
                torch.save({
                    "critic": critic.state_dict(),
                    "critic_target": critic_target.state_dict(),
                }, save_path + f"critic_ckpt_{n_gradient_step + 1}.pt")
                torch.save({
                    "critic": critic.state_dict(),
                    "critic_target": critic_target.state_dict(),
                }, save_path + f"critic_ckpt_latest.pt")

            n_gradient_step += 1
            if n_gradient_step >= args.gradient_steps:
                break

    # ---------------------- Inference ----------------------
    elif args.mode == "inference":

        actor.load(save_path + f"diffusion_ckpt_{args.ckpt}.pt")
        critic_ckpt = torch.load(save_path + f"critic_ckpt_{args.ckpt}.pt")
        critic.load_state_dict(critic_ckpt["critic"])
        critic_target.load_state_dict(critic_ckpt["critic_target"])

        actor.eval()
        critic.eval()
        critic_target.eval()

        env_eval = gym.vector.make(args.task.env_name, args.num_envs)
        normalizer = dataset.get_normalizer()
        episode_rewards = []

        prior = torch.zeros((args.num_envs * args.num_candidates, act_dim), device=args.device)
        for i in range(args.num_episodes):

            obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0

            while not np.all(cum_done) and t < 1000 + 1:
                # normalize obs
                obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
                obs = obs.unsqueeze(1).repeat(1, args.num_candidates, 1).view(-1, obs_dim)

                # sample actions
                act, log = actor.sample(
                    prior,
                    solver=args.solver,
                    n_samples=args.num_envs * args.num_candidates,
                    sample_steps=args.sampling_steps,
                    condition_cfg=obs, w_cfg=1.0,
                    use_ema=args.use_ema, temperature=args.temperature)

                # resample
                with torch.no_grad():
                    q = critic_target.q_min(obs, act)
                    q = q.view(-1, args.num_candidates, 1)
                    w = torch.softmax(q * args.task.weight_temperature, 1)
                    act = act.view(-1, args.num_candidates, act_dim)

                    indices = torch.multinomial(w.squeeze(-1), 1).squeeze(-1)
                    sampled_act = act[torch.arange(act.shape[0]), indices].cpu().numpy()

                # step
                obs, rew, done, info = env_eval.step(sampled_act)

                t += 1
                cum_done = done if cum_done is None else np.logical_or(cum_done, done)
                ep_reward += (rew * (1 - cum_done)) if t < 1000 else rew
                print(f'[t={t}] rew: {np.around((rew * (1 - cum_done)), 2)}')

                if np.all(cum_done):
                    break

            episode_rewards.append(ep_reward)

        episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
        episode_rewards = np.array(episode_rewards).reshape(-1) * 100
        mean = np.mean(episode_rewards)
        err = np.std(episode_rewards) / np.sqrt(len(episode_rewards))
        print(mean, err)

        if args.enable_wandb:
            wandb.log({'Mean Reward': mean, 'Error': err})
            wandb.finish()
            
    elif args.mode == "visual":
        import imageio, mujoco_py, cv2

        actor.load(save_path + f"diffusion_ckpt_{args.ckpt}.pt")
        critic_ckpt = torch.load(save_path + f"critic_ckpt_{args.ckpt}.pt", map_location=args.device)
        critic.load_state_dict(critic_ckpt["critic"])
        critic_target.load_state_dict(critic_ckpt["critic_target"])

        actor.eval()
        critic.eval()
        critic_target.eval()

        env_eval = gym.make(args.task.env_name)
        normalizer = dataset.get_normalizer()
        episode_rewards = []

        prior = torch.zeros((1 * args.num_candidates, act_dim), device=args.device)
        num_fail = num_success = 0
        for i in range(args.num_episodes):

            obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0
            # viewer = env_eval._get_viewer('human')
            camera = mujoco_py.MjRenderContextOffscreen(env_eval.sim, device_id=-1)
            obs = obs[None, :]
            
            frame_cache = []

            while not np.all(cum_done) and t < 1000 + 1:
                camera.cam.lookat[:2] = env_eval.data.qpos[0:2] 
                
                # normalize obs
                obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
                obs = obs.unsqueeze(1).repeat(1, args.num_candidates, 1).view(-1, obs_dim)

                # sample actions
                act, log = actor.sample(
                    prior,
                    solver=args.solver,
                    n_samples=1 * args.num_candidates,
                    sample_steps=args.sampling_steps,
                    condition_cfg=obs, w_cfg=1.0,
                    use_ema=args.use_ema, temperature=args.temperature)

                # resample
                with torch.no_grad():
                    q = critic_target.q_min(obs, act)
                    q = q.view(-1, args.num_candidates, 1)
                    w = torch.softmax(q * args.task.weight_temperature, 1)
                    act = act.view(-1, args.num_candidates, act_dim)

                    indices = torch.multinomial(w.squeeze(-1), 1).squeeze(-1)
                    sampled_act = act[torch.arange(act.shape[0]), indices].cpu().numpy()

                # step
                obs, rew, done, info = env_eval.step(sampled_act)
                obs = obs[None, :]
                image = camera.render(800, 600, -1)
                image = camera.read_pixels(width=800, height=600, depth=False)
                image = cv2.flip(image, 0)
                
                # env_eval.render()
                frame_cache.append(image)
                # writer.append_data(image)

                t += 1
                cum_done = done if cum_done is None else np.logical_or(cum_done, done)
                ep_reward += rew
                # print(f'[t={t}] xy: {obs[:, :2]}')
                # print(f'[t={t}] rew: {ep_reward}')

                if np.all(cum_done):
                    break

            episode_rewards.append(np.clip(ep_reward, 0., 1.))
            if ep_reward == 0:
                print("failure")
                num_fail += 1
                writer = imageio.get_writer(video_path + f"rendering_failure_{num_fail}.mp4", fps=30)
            else:
                print("success")
                num_success += 1
                writer = imageio.get_writer(video_path + f"rendering_success_{num_success}.mp4", fps=30)
            
            for image in frame_cache:
                writer.append_data(image)
            writer.close()

        episode_rewards = [env.get_normalized_score(r) for r in episode_rewards]
        episode_rewards = np.array(episode_rewards).reshape(-1) * 100
        mean = np.mean(episode_rewards)
        err = np.std(episode_rewards) / np.sqrt(len(episode_rewards))
        print(mean, err)

    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    pipeline()
