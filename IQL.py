import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import itertools

EPS = 1e-12


def soft_clamp(x, low, high):
	x = torch.tanh(x)
	x = low + 0.5 * (high - low) * (x + 1)
	return x


class GaussianActor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, hidden=(256, 256), state_dependet_std=False, device_id=-1):
		super(GaussianActor, self).__init__()

		if device_id == -1:
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu")

		self.l1 = nn.Linear(state_dim, hidden[0])
		self.l2 = nn.Linear(hidden[0], hidden[1])
		self.l3_mu = nn.Linear(hidden[1], action_dim)

		if state_dependet_std:
			self.l3_log_std = nn.Linear(hidden[1], action_dim)
		else:
			self.log_std = torch.zeros(action_dim).to(self.device)

		self.log_std_bounds = (-5., 0.)

		self.max_action = max_action
		self.state_dependet_std = state_dependet_std

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		mu = self.max_action * torch.tanh(self.l3_mu(a))
		if self.state_dependet_std:
			log_std = self.max_action * torch.tanh(self.l3_log_std(a))
			log_std = soft_clamp(log_std, *self.log_std_bounds)
		else:
			log_std = self.log_std
		std = log_std.exp()

		dist = D.Normal(mu, std)
		action = dist.rsample()

		return mu, log_std, action

	def get_log_prob(self, state, action):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		mu = self.max_action * torch.tanh(self.l3_mu(a))
		if self.state_dependet_std:
			log_std = self.max_action * torch.tanh(self.l3_log_std(a))
			log_std = soft_clamp(log_std, *self.log_std_bounds)
		else:
			log_std = self.log_std

		log_std = soft_clamp(log_std, *self.log_std_bounds)
		std = log_std.exp()

		dist = D.Normal(mu, std)

		log_prob = dist.log_prob(action)
		if len(log_prob.shape) == 1:
			return log_prob
		else:
			return log_prob.sum(-1, keepdim=True)


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


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

class Value(nn.Module):
	def __init__(self, state_dim):
		super(Value, self).__init__()

		# V architecture
		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

	def forward(self, state ):
		v = F.relu(self.l1(state))
		v = F.relu(self.l2(v))
		v = self.l3(v)
		return v



class IQL(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_freq=1,
		scale=2.0,
		weight_type='',
		expectile=0.9,
		hidden=(256, 256),
		device_id=-1,
		lr=3e-4,
		T_max=1e6
	):
		if device_id == -1:
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu")

		self.actor = GaussianActor(state_dim, action_dim, max_action, hidden=hidden, state_dependet_std=True).to(self.device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
		self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_optimizer, T_max=int(T_max))

		self.critic = Critic(state_dim, action_dim).to(self.device)
		self.value = Value(state_dim).to(self.device)

		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
		self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_freq = policy_freq

		self.scale = scale
		self.weight_type = weight_type
		self.expectile = expectile

		self.action_dim = action_dim
		self.state_dim = state_dim

		self.total_it = 0

		print('scale', self.scale)
		print('hidden', hidden)
		print('weight_type', weight_type)

	def select_action(self, state, stochastic=False):

		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		mu, log_std, action = self.actor(state)
		if stochastic:
			return action.cpu().data.numpy().flatten()

		return mu.cpu().data.numpy().flatten()

	def train(self, replay_buffer, batch_size=256, envname=''):
		self.total_it += 1
		self.critic.train()

		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		if 'antmaze' in envname:
			reward = reward - 1

		# update value
		# Get current Q estimates
		with torch.no_grad():
			current_Q1, current_Q2 = self.critic_target(state, action)
			minQ = torch.min(current_Q1, current_Q2)

		v = self.value(state)
		value_loss = self.expectile_loss(v, minQ, expectile=self.expectile)

		# Optimize the critic
		self.value_optimizer.zero_grad()
		value_loss.backward()
		self.value_optimizer.step()

		with torch.no_grad():
			next_value = self.value(next_state)
			target_Q = reward + not_done * self.discount * next_value

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
				# sa = torch.cat([state, action], 1)
				Q1, Q2 = self.critic_target(state, action)
				Qmin = torch.min(Q1, Q2)

				v = self.value(state)
				adv = Qmin - v

				width = torch.max(adv).detach() - torch.min(adv).detach()

				weight = None
				if self.scale == 0.0:
					weight = adv
				elif self.weight_type == 'clamp':
					weight = torch.exp(self.scale * (adv)).clamp(0, 100)
				else:
					weight = torch.exp(self.scale * (adv - adv.max()) / width)

			# Compute actor loss
			log_p = self.actor.get_log_prob(state, action)
			actor_loss = - (log_p * weight.detach() ).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()
			self.actor_scheduler.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
				
		return critic_loss.item()

	def expectile_loss(self, x, y, expectile):
		diff = y - x
		weight = (torch.ones(size=(x.shape[0], 1)) * (1 - expectile)).to(self.device)
		ind = (diff > 0)
		weight[ind] = expectile
		# print('diff', diff[0:10])
		# print('weight', weight[0:10])

		return (weight * (diff ** 2)).mean()

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")



	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.actor.load_state_dict(torch.load(filename + "_actor"))
