import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import itertools

EPS = 1e-12

class LatentGaussianActor(nn.Module):
	def __init__(self, state_dim, action_dim, latent_dim, max_action, hidden=(256, 256)):
		super(LatentGaussianActor, self).__init__()
		self.l1_latent = nn.Linear(latent_dim, state_dim)
		self.l1 = nn.Linear(2 * state_dim, hidden[0])
		self.l2 = nn.Linear(hidden[0], hidden[1])
		self.l3_mu = nn.Linear(hidden[1], action_dim)
		self.l3_log_std = nn.Linear(hidden[1], action_dim)

		self.log_std_bounds = (-5., 0.)

		self.max_action = max_action

	def forward(self, state, latent):
		z = F.relu(self.l1_latent(latent))
		sz = torch.cat([state, z], 1)
		a = F.relu(self.l1(sz))
		a = F.relu(self.l2(a))
		mu = self.max_action * torch.tanh(self.l3_mu(a))
		log_std = self.max_action * torch.tanh(self.l3_log_std(a))

		log_std = soft_clamp(log_std, *self.log_std_bounds)
		std = log_std.exp()

		dist = D.Normal(mu, std)
		action = dist.rsample()

		return mu, log_std, action

	def get_log_prob(self, state, action, latent):
		z = F.relu(self.l1_latent(latent))
		sz = torch.cat([state, z], 1)
		a = F.relu(self.l1(sz))
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


def soft_clamp(x, low, high):
	x = torch.tanh(x)
	x = low + 0.5 * (high - low) * (x + 1)
	return x


class GaussianMixtureActor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, component_num, hidden=(256, 256), temperature=.67, device_id=-1):
		super(GaussianMixtureActor, self).__init__()

		if device_id == -1:
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu")

		self.l1 = nn.Linear(state_dim, hidden[0])
		self.l2 = nn.Linear(hidden[0], hidden[1])
		self.l3_w = nn.Linear(hidden[1], component_num)

		self.component_policy = LatentGaussianActor(state_dim=state_dim, action_dim=action_dim,
													latent_dim=component_num,
													max_action=max_action, hidden=hidden)

		self.max_action = max_action
		self.component_num = component_num
		self.temperature = temperature

	def forward(self, state, pri=False):
		h = F.relu(self.l1(state))
		h = F.relu(self.l2(h))
		h = self.l3_w(h)
		alpha = torch.softmax(h,dim=1)
		# print('h', h.shape)
		# print('alpha', alpha.shape)

		z_disc = self.gumbel_softmax(alpha, hard=True)
		if pri:
			print('z_disc', z_disc)
			print('alpha', alpha)

		mu, log_std, action = self.component_policy(state, z_disc.detach())

		return mu, log_std, action

	def get_log_prob(self, state, action):
		w = F.relu(self.l1(state))
		w = F.relu(self.l2(w))
		alpha = torch.softmax(self.l3_w(w), dim=1)

		z_int = np.arange(0, self.component_num)
		latent_sample = to_one_hot(z_int, self.component_num)

		latent_sample = torch.FloatTensor(latent_sample).to(self.device)
		latent_tile = (latent_sample.repeat(1, state.shape[0])).view(-1, self.component_num)

		state_tile = state.repeat(self.component_num, 1)
		action_tile = action.repeat(self.component_num, 1)

		logp_tile = self.component_policy.get_log_prob(state_tile, action_tile, latent_tile)
		logp_tile = torch.t(logp_tile.reshape(-1, state.shape[0]))

		log_p = torch.log(torch.sum(alpha * torch.exp(logp_tile), dim=1, keepdim=True) + EPS)

		# print('alpha', alpha.shape)
		# print('logp_tile', logp_tile.shape)
		# print('alpha', alpha[0:5])
		# print('logp_tile', logp_tile[0:5])
		# print('latent_tile', latent_tile[0:5])
		# print('alpha * torch.exp(logp_tile)', (alpha * torch.exp(logp_tile)).shape)

		# print('log_p', log_p.shape)
		# print('log_p', log_p[0:5])

		return log_p

	def gumbel_softmax(self, alpha, hard=False):
		"""
		Samples from a gumbel-softmax distribution using the reparameterization
		trick.
		Parameters
		----------
		alpha : torch.Tensor
			Parameters of the gumbel-softmax distribution. Shape (N, D)
		----------
		adopted from https://github.com/Schlumberger/joint-vae/
		"""
		# Sample from gumbel distribution
		y = self.gumbel_softmax_sample(alpha)
		if not hard:
			return y
		else:
			shape = y.size()
			_, max_alpha = torch.max(alpha, dim=1)
			one_hot_samples = torch.zeros(alpha.size()).to(self.device)
			one_hot_samples.scatter_(1, max_alpha.view(-1, 1), 1)

			y_hard = one_hot_samples.view(*shape)
			# Set gradients w.r.t. y_hard gradients w.r.t. y
			y_hard = (y_hard - y).detach() + y

			return y_hard

	def gumbel_softmax_sample(self, alpha):
		unif = torch.rand(alpha.size()).to(self.device)
		gumbel = -torch.log(-torch.log(unif + EPS) + EPS)
		# Reparameterize to create gumbel softmax sample
		log_alpha = torch.log(alpha + EPS)
		logit = (log_alpha + gumbel) / self.temperature
		return F.softmax(logit, dim=1)

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


def to_one_hot(y, class_num):
	one_hot = np.zeros((y.shape[0], class_num))
	one_hot[range(y.shape[0]), y] = 1.0

	return one_hot


class mixAWAC(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		component_num,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_freq=2,
		scale=2.0,
		weight_type='',
		hidden=(256, 256),
		device_id=-1,
		lr=3e-4,
		T_max=1e6,
		schedule=None,
	):
		if device_id == -1:
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu")

		self.actor = GaussianMixtureActor(state_dim, action_dim, max_action, component_num, hidden=hidden).to(self.device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
		if schedule is not None:
			self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_optimizer,
																			  T_max=int(T_max / policy_freq))

		self.critic = Critic(state_dim, action_dim).to(self.device)

		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_freq = policy_freq

		self.scale = scale
		self.weight_type = weight_type
		self.schedule = schedule

		self.action_dim = action_dim
		self.state_dim = state_dim

		self.total_it = 0

		print('scale', self.scale)
		print('hidden', hidden)
		print('schedule', schedule)
		print('mixAWAC')


	def select_action(self, state, stochastic=False, pri=False):

		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		mu, log_std, action = self.actor(state, pri=pri)
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

		with torch.no_grad():
			_, _, next_action = self.actor_target(next_state)
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

		if self.total_it % 1000 == 0:
			print("critic loss", critic_loss.item())

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:
			with torch.no_grad():

				Q1, Q2 = self.critic_target(state, action)
				Qmin = torch.min(Q1, Q2)

				_, _, action_pred = self.actor_target(state)
				Q1_pred, Q2_pred = self.critic_target(state, action_pred)
				v = torch.min(Q1_pred, Q2_pred)
				adv = Qmin - v

				width = torch.max(adv).detach() - torch.min(adv).detach()

				weight = None
				if self.scale == 0.0:
					weight = adv
				elif self.weight_type == 'clamp':
					weight = torch.exp(self.scale * (adv)).clamp(0, 100)
				elif self.weight_type == 'adapt3':
					weight = torch.exp(self.scale * (adv - adv.min()) / width)
				else:
					weight = torch.exp(self.scale * (adv - adv.max()) / width)

			# Compute actor loss
			log_p = self.actor.get_log_prob(state, action)
			actor_loss = - (log_p * weight.detach()).mean()

			# Optimize the actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			if self.schedule is not None:
				self.actor_scheduler.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		return critic_loss.item()

	def save(self, filename):
		torch.save(self.actor.state_dict(), filename + "_actor")

	def load(self, filename):
		self.actor.load_state_dict(torch.load(filename + "_actor"))