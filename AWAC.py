import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import itertools

EPS = 1e-12

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, latent_dim, max_action, hidden=(256, 256)):
		super(Actor, self).__init__()
		self.l1_latent = nn.Linear(latent_dim, state_dim)
		self.l1 = nn.Linear(2*state_dim, hidden[0])
		self.l2 = nn.Linear(hidden[0], hidden[1])
		self.l3 = nn.Linear(hidden[1], action_dim)
		
		self.max_action = max_action

	def forward(self, state, latent):
		z = F.relu(self.l1_latent(latent))
		sz = torch.cat([state, z], 1)
		a = F.relu(self.l1(sz))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


def soft_clamp(x, low, high):
	x = torch.tanh(x)
	x = low + 0.5 * (high - low) * (x + 1)
	return x


class GaussianActor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, hidden=(256, 256)):
		super(GaussianActor, self).__init__()

		self.l1 = nn.Linear(state_dim, hidden[0])
		self.l2 = nn.Linear(hidden[0], hidden[1])
		self.l3_mu = nn.Linear(hidden[1], action_dim)
		self.l3_log_std = nn.Linear(hidden[1], action_dim)

		self.log_std_bounds = (-5., 0.)
		self.mu_bounds = (-1., 1.)

		self.max_action = max_action

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		mu = self.max_action * torch.tanh(self.l3_mu(a))
		log_std = self.max_action * torch.tanh(self.l3_log_std(a))
		log_std = soft_clamp(log_std, *self.log_std_bounds)

		std = log_std.exp()
		dist = D.Normal(mu, std)

		action = dist.rsample()

		return action, mu, std

	def get_log_prob(self, state, action):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		mu = self.max_action * torch.tanh(self.l3_mu(a))
		log_std = self.max_action * torch.tanh(self.l3_log_std(a))
		log_std = soft_clamp(log_std, *self.log_std_bounds)

		std = log_std.exp()
		dist = D.Normal(mu, std)

		log_prob = dist.log_prob(action)
		if len(log_prob.shape) == 1:
			return log_prob
		else:
			return log_prob.sum(-1, keepdim=True)


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, hidden=(256,256)):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, hidden[0])
		self.l2 = nn.Linear(hidden[0], hidden[1])
		self.l3 = nn.Linear(hidden[1], 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, hidden[0])
		self.l5 = nn.Linear(hidden[0], hidden[1])
		self.l6 = nn.Linear(hidden[1], 1)

	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2

	def Q1(self, state, action):

		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		return q1


class AWAC(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_freq=2,
		scale=2.0,
		score_type='adv',
		hidden=(256, 256),
		weight_type='clamp',
		device_id=-1,
		lr=3e-4
	):
		if device_id == -1:
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu")

		self.actor = GaussianActor(state_dim, action_dim, max_action, hidden=hidden).to(self.device)
		self.actor_target = copy.deepcopy(self.actor)
		self.behavior = GaussianActor(state_dim, action_dim, max_action, hidden=hidden).to(self.device)

		# self.posterior_optimizer = torch.optim.Adam(self.posterior.parameters(), lr=3e-4)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
		self.behavior_optimizer = torch.optim.Adam(self.behavior.parameters(), lr=lr)

		self.critic = Critic(state_dim, action_dim, hidden=hidden).to(self.device)

		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_freq = policy_freq

		self.scale = scale
		self.score_type = score_type
		self.weight_type = weight_type

		self.action_dim = action_dim

		self.total_it = 0

		self.behavior_it = 0
		self.q_it = 0
		self.pi_it = 0

		print('AWAC')
		print('scale', self.scale)
		print('score_type', self.score_type)
		print('weight_type', self.weight_type)
		print('hidden', hidden)

	def select_action(self, state, stochastic=False): # only for continuous latent variables
		self.critic.eval()
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		action, mu, _ = self.actor(state)
		if stochastic:
			return action.cpu().data.numpy().flatten()
		return mu.cpu().data.numpy().flatten()

	def train_behavior(self, replay_buffer, batch_size=256):
		self.behavior_it += 1

		# Sample replay buffer
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		# Compute actor loss
		log_prob = self.behavior.get_log_prob(state, action)
		behavior_loss = (-log_prob).mean()

		# Optimize the critic
		self.behavior_optimizer.zero_grad()
		behavior_loss.backward()
		self.behavior_optimizer.step()

		return behavior_loss.item()

	def train(self, replay_buffer, batch_size=256, envname='', sample_num=5):
		self.total_it += 1
		self.critic.train()

		# Sample replay buffer
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		if 'antmaze' in envname:
			reward = reward - 1

		with torch.no_grad():
			next_action, _, _ = self.actor(next_state)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		if self.total_it %1000 ==0:
			print("critic loss", critic_loss.item())

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:
			with torch.no_grad():
				sa = torch.cat([state, action], 1)
				Q1, Q2 = self.critic(state, action)
				Qmin = torch.min(Q1, Q2)

				if self.score_type == 'adv':
					state_tile = state.repeat(sample_num, 1)
					action_tile, _, _ = self.actor_target(state_tile)

					Q1_base, Q2_base = self.critic(state_tile, action_tile)
					minQ_base = torch.min(Q1_base, Q2_base)

					value = torch.mean(torch.t(minQ_base.reshape(sample_num, state.shape[0])), dim=1, keepdim=True)
					adv = Qmin - value

			if self.score_type == 'adv':
				score = adv
			else:
				score = Qmin

			weight = None
			if self.scale == 0.0:
				weight = score
			else:
				if self.weight_type == 'clamp':
					weight = torch.exp(self.scale * score).clamp(0, 100)
				elif self.weight_type == 'adapt':
					with torch.no_grad():
						width = torch.max(score) - torch.min(score)

					weight = torch.exp(self.scale * (score - torch.max(score).detach()) / width.detach())
					weight = weight / weight.mean()
				else:
					weight = torch.exp(self.scale * (score - score.max()))
					weight = weight / weight.mean()

			log_p = self.actor.get_log_prob(state, action)

			# Compute actor loss
			actor_loss = (- weight * log_p).mean()

			# Optimize the actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			if self.total_it % self.policy_freq == 0:
				# Update the frozen target models
				for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		return critic_loss.item()

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.actor.state_dict(), filename + "_actor")

	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.actor.load_state_dict(torch.load(filename + "_actor"))
