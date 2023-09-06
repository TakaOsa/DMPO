import numpy as np
import torch
# import adept_envs
import gym
import argparse
import os
import d4rl

import utils
import DMPO, AWAC, mixAWAC, IQL, LP_AWAC, DMPO_ant

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, mean, std,
				seed_offset=100, eval_episodes=10, render=False):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + seed_offset)

	avg_reward = 0.
	episode_return_set = []
	episode_d4rl_set = []
	for _ in range(eval_episodes):
		episode_return = 0
		state, done = eval_env.reset(), False
		while not done:
			if render:
				eval_env.render()
			state = (np.array(state).reshape(1,-1) - mean)/std
			action = policy.select_action(state)
			# action = (np.array(action).reshape(1, -1) * a_std + a_mean).flatten()
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward
			episode_return += reward

		episode_return_set.append(episode_return)
		episode_d4rl_set.append(eval_env.get_normalized_score(episode_return))

	avg_reward /= eval_episodes
	d4rl_score = eval_env.get_normalized_score(avg_reward)

	print("----------------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes d4rl_score: {d4rl_score:.3f} return: {avg_reward:.1f}")
	print("----------------------------------------------")
	return d4rl_score, avg_reward, np.asarray(episode_d4rl_set), np.asarray(episode_return_set)


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	# Experiment
	parser.add_argument("--policy", default="DMPO_ant")               # Policy name
	parser.add_argument("--env", default="antmaze-large-diverse-v0")   # D4RL environment name kitchen-complete-v0, hammer-human-v0,hopper-medium-v0, walker2d-medium-replay-v2, antmaze-large-play-v0
	parser.add_argument("--seed", default=1, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--c_loss_freq", default=5e3, type=int)  	# How often (time steps) we record the critic loss
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--save_model", default=True, type=bool)    # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name

	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--normalize", default=True)                # flag for state normalization

	parser.add_argument("--scale", default=10.0, type=float)
	parser.add_argument("--score_type", default='adv')
	parser.add_argument("--hidden", default=(256, 256))
	parser.add_argument("--weight_type", default='')
	parser.add_argument("--device_id", default=-1, type=int)

	# DMPO
	parser.add_argument("--latent_disc_dim", default=8, type=int)
	parser.add_argument("--info-reg", action='store_true', default=False)
	parser.add_argument("--ilr", default=5e-7, type=float)
	parser.add_argument("--doubleq_min", default=0.7, type=float) # for Ant

	#IQL
	parser.add_argument("--expectile", default=0.9, type=float)

	# LP-AWAC
	parser.add_argument('--max_latent_action', default=2.0, type=float)  # maximum value for the latent policy
	parser.add_argument('--no_noise', action='store_true')


	args = parser.parse_args()

	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")
	if not os.path.exists("./results/" + args.policy):
		os.makedirs("./results/" + args.policy)

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env = gym.make(args.env)

	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])

	state_norm = ''
	info_reg = ''
	if args.normalize == False:
		state_norm = '_no_state_normal'
	if args.info_reg:
		info_reg = '_info_reg' + '_ilr' + str(args.ilr)

	file_name = f"{args.policy}_test_{args.env}_{args.seed}"
	if args.policy == 'DMPO':
		file_name = args.policy + '_disc' + str(args.latent_disc_dim) + state_norm \
					+ '_scale' + str(args.scale) + '_' + args.weight_type + info_reg  \
					+ f"_{args.env}_{args.seed}"

	elif args.policy == 'AWAC':
		file_name = args.policy + '_hidden' + str(args.hidden) + state_norm + \
					'_scale' + str(args.scale) + '_' + args.weight_type + f"_{args.env}_{args.seed}"

	elif args.policy == 'IQL':
		file_name = args.policy + '_expectile' + str(args.expectile) + state_norm  \
					+ '_scale' + str(args.scale) + '_' + args.weight_type  + f"_{args.env}_{args.seed}"

	elif args.policy == 'mixAWAC':
		file_name = args.policy + '_comp' + str(args.latent_disc_dim) \
					+ state_norm + '_' + args.weight_type + '_scale' + str(args.scale) + f"_{args.env}_{args.seed}"

	elif args.policy == 'DMPO_ant':
		file_name = args.policy + '_disc' + str(args.latent_disc_dim) + state_norm  \
					+ '_scale' + str(args.scale) + '_' + args.weight_type + info_reg + '_dq_min' + str(args.doubleq_min) \
					+ f"_{args.env}_{args.seed}"

	# Initialize policy
	if args.policy == 'AWAC':
		policy = AWAC.AWAC(state_dim=state_dim, action_dim=action_dim,
							max_action=max_action,
							discount=args.discount, policy_freq=args.policy_freq, weight_type=args.weight_type,
							scale=args.scale, score_type=args.score_type,
							hidden=args.hidden, device_id=args.device_id)

	elif args.policy == 'IQL':
		policy = IQL.IQL(state_dim=state_dim, action_dim=action_dim,
							 max_action=max_action, discount=args.discount,
							 expectile=args.expectile, policy_freq=args.policy_freq, weight_type=args.weight_type,
							 scale=args.scale, hidden=args.hidden, device_id=args.device_id)

	elif args.policy == 'mixAWAC':
		policy = mixAWAC.mixAWAC(state_dim=state_dim, action_dim=action_dim, component_num=args.latent_disc_dim,
						   max_action=max_action, discount=args.discount,
						   policy_freq=args.policy_freq, scale=args.scale, weight_type=args.weight_type,
						   device_id=args.device_id, hidden=args.hidden)

	elif args.policy == 'LP_AWAC':
		latent_dim = action_dim * 2
		policy = LP_AWAC.LP_AWAC(state_dim, action_dim, latent_dim, max_action,
							  discount=args.discount, tau=args.tau,
							  max_latent_action=args.max_latent_action,  no_noise=args.no_noise)

	elif args.policy == 'DMPO_ant':
		policy = DMPO_ant.DMPO_ant(state_dim=state_dim, action_dim=action_dim,
						   latent_disc_dim=args.latent_disc_dim, scale=args.scale, weight_type=args.weight_type,
						   max_action=max_action, device_id=args.device_id,
						   discount=args.discount, policy_freq=args.policy_freq, info_reg=args.info_reg, ilr=args.ilr,
						   doubleq_min=args.doubleq_min)
	else: # default
		policy = DMPO.DMPO(state_dim=state_dim, action_dim=action_dim,
						   latent_disc_dim=args.latent_disc_dim,
						   max_action=max_action,
						   discount=args.discount, policy_freq=args.policy_freq, weight_type=args.weight_type,
						   scale=args.scale, hidden=args.hidden, info_reg=args.info_reg, ilr=args.ilr,
						   device_id=args.device_id)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device_id=args.device_id)
	replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))

	if args.normalize:
		mean,std = replay_buffer.normalize_states() 
	else:
		mean,std = 0,1

	print('datasize: ', replay_buffer.size)
	
	evaluations_d4rl= []
	evaluations_return = []
	evaluations_d4rl_epi = []
	evaluations_return_epi = []

	critic_loss_log = []

	c_loss_i = 0
	for t in range(int(args.max_timesteps)):
		c_loss_t = policy.train(replay_buffer, args.batch_size, envname=args.env)
		c_loss_i += c_loss_t / args.c_loss_freq
		if (t+1) % args.c_loss_freq == 0:
			critic_loss_log.append(c_loss_i)
			c_loss_i = 0
			np.savetxt(f"./results/{args.policy}/{file_name}_critic_loss_log.txt", critic_loss_log)

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			print(f"Time steps: {t+1}")
			if 'antmaze' in args.env and t+1 == args.max_timesteps:
				d4rl_score, ave_return, d4l_score_epi, return_epi = \
					eval_policy(policy, args.env, args.seed, mean, std, eval_episodes=100)
			else:
				d4rl_score, ave_return, d4l_score_epi, return_epi = \
					eval_policy(policy, args.env, args.seed, mean, std)

			evaluations_d4rl.append(d4rl_score)
			evaluations_return.append(ave_return)
			evaluations_d4rl_epi.append(d4l_score_epi)
			evaluations_return_epi.append(return_epi)
			np.savetxt(f"./results/{args.policy}/{file_name}_d4rl.txt", evaluations_d4rl)
			np.savetxt(f"./results/{args.policy}/{file_name}_return.txt", evaluations_return)
			if args.save_model: policy.save(f"./models/{file_name}")
