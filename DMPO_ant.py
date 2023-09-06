import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import itertools

EPS = 1e-12

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, latent_dim, max_action, hidden=(256, 256,256)):
		super(Actor, self).__init__()
		self.l1_latent = nn.Linear(latent_dim, state_dim)
		self.l1 = nn.Linear(2*state_dim, hidden[0])
		self.l2 = nn.Linear(hidden[0], hidden[1])
		self.l3 = nn.Linear(hidden[1], hidden[2])
		self.l4 = nn.Linear(hidden[2], action_dim)
		
		self.max_action = max_action

	def forward(self, state, latent):
		z = F.relu(self.l1_latent(latent))
		sz = torch.cat([state, z], 1)
		a = F.relu(self.l1(sz))
		a = F.relu(self.l2(a))
		a = F.relu(self.l3(a))
		return self.max_action * torch.tanh(self.l4(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, hidden=(256, 256, 256)):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, hidden[0])
		self.l2 = nn.Linear(hidden[0], hidden[1])
		self.l3 = nn.Linear(hidden[1], hidden[2])
		self.l4 = nn.Linear(hidden[2], 1)

		self.l5 = nn.Linear(state_dim + action_dim, hidden[0])
		self.l6 = nn.Linear(hidden[0], hidden[1])
		self.l7 = nn.Linear(hidden[1], hidden[2])
		self.l8 = nn.Linear(hidden[2], 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = F.relu(self.l3(q1))
		q1 = (self.l4(q1))

		q2 = F.relu(self.l5(sa))
		q2 = F.relu(self.l6(q2))
		q2 = F.relu(self.l7(q2))
		q2 = (self.l8(q2))

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
		self.l3 = nn.Linear(256, 256)
		self.l4 = nn.Linear(256, 1)

	def forward(self, state ):
		v = F.relu(self.l1(state))
		v = F.relu(self.l2(v))
		v = F.relu(self.l3(v))
		v = self.l4(v)
		return v

class PosteriorApproximator(nn.Module):
	def __init__(self, state_dim, action_dim, latent_disc_dim, temperature=.67, hidden=(256, 256, 256), device_id=-1):
		super(PosteriorApproximator, self).__init__()

		# Encoder architecture
		self.l1 = nn.Linear(state_dim + action_dim, hidden[0])
		self.l2 = nn.Linear(hidden[0], hidden[1])
		self.l3 = nn.Linear(hidden[1], hidden[2])

		self.l4_disc = nn.Linear(hidden[1], latent_disc_dim)
		self.latent_disc_dim = latent_disc_dim

		self.temperature = temperature

		if device_id == -1:
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu")

	def sample_gumbel_softmax(self, alpha):
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
		if self.training:
			# Sample from gumbel distribution
			unif = torch.rand(alpha.size()).to(self.device)
			gumbel = -torch.log(-torch.log(unif + EPS) + EPS)
			# Reparameterize to create gumbel softmax sample
			log_alpha = torch.log(alpha + EPS)
			logit = (log_alpha + gumbel) / self.temperature
			return F.softmax(logit, dim=1)
		else:
			# In reconstruction mode, pick most likely sample
			_, max_alpha = torch.max(alpha, dim=1)
			one_hot_samples = torch.zeros(alpha.size())
			# On axis 1 of one_hot_samples, scatter the value 1 at indices
			# max_alpha. Note the view is because scatter_ only accepts 2D
			# tensors.
			one_hot_samples.scatter_(1, max_alpha.view(-1, 1).data.cpu(), 1).to(self.device)
			return one_hot_samples

	def forward(self, state, action):
		z_disc, alpha = self.encode(state, action)

		return z_disc, alpha

	def encode(self, state, action):
		sa = torch.cat([state, action], 1)
		h = F.relu(self.l1(sa))
		h = F.relu(self.l2(h))
		h = F.relu(self.l3(h))

		alpha = F.softmax(self.l4_disc(h), dim=1)
		z_disc = self.sample_gumbel_softmax(alpha)

		return z_disc, alpha


def to_one_hot(y, class_num):
	one_hot = np.zeros((y.shape[0], class_num))
	one_hot[range(y.shape[0]), y] = 1.0

	return one_hot


def uniform_sampling(cont_dim, disc_dim, sample_num):
	z = None
	z_cont = None
	z_disc = None
	if cont_dim > 0:
		z_cont = np.random.uniform(-1, 1, size=(sample_num, cont_dim))
		if disc_dim < 1:
			z = z_cont
	if disc_dim > 0:
		z_int = np.random.randint(0, disc_dim, sample_num)
		z_disc = to_one_hot(z_int, disc_dim)
		if cont_dim < 1:
			z = z_disc
		else:
			z = np.hstack([z_cont, z_disc])

	return z, z_cont, z_disc


class DMPO_ant(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		latent_disc_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		clip=0.5,
		beta=1.0,
		hidden=(256, 256, 256),
		device_id=-1,
		lr=2e-4,
		doubleq_min = 0.7,
		scale = 10,
		weight_type='clamp',
		info_reg=False,
		ilr=5e-7
	):
		print('torch.cuda.is_available()', torch.cuda.is_available())

		if device_id == -1:
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu")

		self.latent_dim = latent_disc_dim

		self.actor = Actor(state_dim, action_dim, self.latent_dim, max_action, hidden=hidden).to(self.device)
		self.actor_target = copy.deepcopy(self.actor)

		self.posterior = PosteriorApproximator(state_dim, action_dim, latent_disc_dim, hidden=hidden, device_id=device_id).to(self.device)

		self.actor_optimizer = torch.optim.Adam(itertools.chain(self.posterior.parameters(), self.actor.parameters()), lr=lr)

		if info_reg:
			self.info_posterior = PosteriorApproximator(state_dim, action_dim, latent_disc_dim, hidden=hidden,
														device_id=device_id).to(self.device)
			self.info_optimizer = torch.optim.Adam(
				itertools.chain(self.actor.parameters(), self.info_posterior.parameters()),
				lr=ilr)

		self.critic = Critic(state_dim, action_dim).to(self.device)
		self.value = Value(state_dim).to(self.device)
		self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr)

		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.scale = scale
		self.weight_type = weight_type

		self.latent_disc_dim = latent_disc_dim

		self.clip = clip
		self.beta = beta

		self.action_dim = action_dim
		self.state_dim = state_dim

		self.total_it = 0
		self.min_v = None
		self.max_v = None
		self.doubleq_min = doubleq_min
		self.info_reg = info_reg

		self.behavior_it = 0
		self.q_it = 0
		self.pi_it = 0

		print('DMPO_ant')
		print('scale', self.scale)
		print('latent_disc_dim', self.latent_disc_dim)
		print('hidden', hidden)
		print('doubleq_min', doubleq_min)
		print('weight_type', self.weight_type)
		print('info_reg', self.info_reg)
		print('ilr', ilr)

	def select_action(self, state):
		self.posterior.eval()
		self.critic.eval()
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		latent, _ = self.select_latent(state, self.actor)
		self.critic.train()
		self.posterior.train()
		return self.actor(state, latent).cpu().data.numpy().flatten()

	def select_action_latent(self, state, latent):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		latent = torch.FloatTensor(latent.reshape(1, -1)).to(self.device)
		return self.actor(state, latent).cpu().data.numpy().flatten()

	def select_latent(self, state, policy):
		eval_num = self.latent_disc_dim
		z_int = np.arange(0, self.latent_disc_dim)
		latent_sample = to_one_hot(z_int, self.latent_disc_dim)

		latent_sample = torch.FloatTensor(latent_sample).to(self.device)
		latent_tile = (latent_sample.repeat(1, state.shape[0])).view(-1, self.latent_disc_dim)
		state_tile = state.repeat(eval_num, 1)

		action_tile = policy(state_tile, latent_tile)

		Q1_pred, Q2_pred = self.critic(state_tile, action_tile)
		# Qvalue = torch.min(Q1_pred, Q2_pred)
		Qvalue = torch.min(Q1_pred, Q2_pred) * self.doubleq_min \
			   + torch.max(Q1_pred, Q2_pred) * (1 - self.doubleq_min)

		ind = torch.argmax(torch.t(Qvalue.reshape(-1, state.shape[0])), dim=1)
		latent_set = latent_tile.view(eval_num, state.shape[0],self.latent_dim).permute(1,0,2)
		latent_max = latent_set[torch.arange(state.size(0)), ind, :]

		return latent_max, torch.t(Qvalue.reshape(-1, state.shape[0]))[torch.arange(state.size(0)), ind].view(-1,1)

	def sample_action(self, state):
		latent_max, _ = self.select_latent(state, self.actor)
		action = self.actor(state, latent_max)

		return action

	def sample_latent(self, replay_buffer, batch_size=256):
		# Sample replay buffer
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
		# self.posterior.eval()
		latent_sample, _ = self.posterior.encode(state, action)
		Q1, _ = self.critic(state, action)
		_, v = self.select_latent(state, self.actor)

		return latent_sample.cpu().data.numpy(), state.cpu().data.numpy(), action.cpu().data.numpy(), (Q1 - v).cpu().data.numpy()

	def eval_latent(self, state, numpy=True):
		if numpy:
			state = torch.FloatTensor(state.reshape(-1, self.state_dim)).to(self.device)

		eval_num = self.latent_disc_dim
		z_int = np.arange(0, self.latent_disc_dim)
		latent_sample = to_one_hot(z_int, self.latent_disc_dim)

		latent_sample = torch.FloatTensor(latent_sample).to(self.device)
		latent_tile = (latent_sample.repeat(1, state.shape[0])).view(-1, self.latent_disc_dim)
		state_tile = state.repeat(eval_num, 1)
		# print('latent_tile', latent_tile)

		action_tile = self.actor(state_tile, latent_tile)

		Q1_pred, Q2_pred = self.critic(state_tile, action_tile)
		Qvalue = torch.min(Q1_pred, Q2_pred) * self.doubleq_min \
				 + torch.max(Q1_pred, Q2_pred) * (1 - self.doubleq_min)

		if numpy:
			return torch.t(Qvalue.reshape(-1, state.shape[0])).cpu().data.numpy()
		else:
			return torch.t(Qvalue.reshape(-1, state.shape[0]))

	def train(self, replay_buffer, batch_size=256, envname=''):
		self.total_it += 1
		self.posterior.train()
		self.critic.train()

		if self.min_v is None or self.max_v is None:
			if 'antmaze' in envname:
				self.min_v = 0
				self.max_v = 100
			else:
				self.min_v = replay_buffer.reward.min()/ (1 - self.discount)
				self.max_v = replay_buffer.reward.max()/ (1 - self.discount)

			print('min_v', self.min_v)
			print('max_v', self.max_v)

		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		if 'antmaze' in envname:
			reward = reward*100

		with torch.no_grad():
			latent_max, _ = self.select_latent(state, self.actor_target)
			action_policy = self.actor_target(state, latent_max)
			target_Q1, target_Q2 = self.critic_target(state, action_policy)
			target_V = torch.min(target_Q1, target_Q2) * self.doubleq_min \
					   + torch.max(target_Q1, target_Q2) * (1 - self.doubleq_min)

		v = self.value(state)
		value_loss = F.mse_loss(v, target_V.clamp(self.min_v, self.max_v))

		# Optimize the critic
		self.value_optimizer.zero_grad()
		value_loss.backward()
		self.value_optimizer.step()

		with torch.no_grad():
			target_v = self.value(next_state)
			target_Q = reward + not_done * self.discount * target_v

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		if self.total_it %1000 ==0:
			print(self.total_it, ": critic loss", critic_loss.item())

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:
			with torch.no_grad():
				Q1, Q2 = self.critic(state, action)
				Qvalue = torch.min(Q1, Q2) * self.doubleq_min \
							 + torch.max(Q1, Q2) * (1 - self.doubleq_min)

				state_value = self.value(state)
				adv = Qvalue - state_value

				width = torch.max(adv).detach() - torch.min(adv).detach()

				weight = None
				if self.weight_type == 'clamp':
					weight = torch.exp(self.scale * (adv)).clamp(0, 100)
				else:
					weight = torch.exp(self.scale * (adv - adv.max()) / width)
					weight = weight / weight.mean()

			# Compute mutual information loss
			if self.info_reg:
				label_latent = torch.randint(low=0, high=self.latent_disc_dim - 1,
											 size=(state.shape[0],)).to(self.device)
				latent_disc_sample = F.one_hot(label_latent,
											   num_classes=self.latent_disc_dim).type(torch.FloatTensor).to(
					self.device)

				action_pred_info = self.actor(state, latent_disc_sample.detach())
				_, alpha_info = self.info_posterior(state, action_pred_info)

				cross_entropy_loss = nn.CrossEntropyLoss()
				info_loss = cross_entropy_loss(alpha_info, label_latent.detach())

				self.info_optimizer.zero_grad()
				info_loss.backward()
				self.info_optimizer.step()

			# Compute actor loss
			latent_pred, alpha = self.posterior(state, action)

			q_latent = self.eval_latent(state, numpy=False)
			q_latent = F.softmax(q_latent, dim=1)

			kl_loss = self.weighted_kl(weight.detach(), alpha, q_latent.detach())

			action_pred = self.actor(state, latent_pred)
			actor_loss = self.weighted_mse_loss(weight.detach(), action_pred, action)*self.action_dim + self.beta * kl_loss
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		return critic_loss.item()

	def kl_discrete_loss(self, alpha):
		"""
		Calculates the KL divergence between a categorical distribution and a
		uniform categorical distribution.
		Parameters
		----------
		alpha : torch.Tensor
			Parameters of the categorical or gumbel-softmax distribution.
			Shape (N, D)
		"""
		disc_dim = int(alpha.size()[-1])
		log_dim = torch.Tensor([np.log(disc_dim)]).to(self.device)
		# Calculate negative entropy of each row
		neg_entropy = torch.sum(alpha * torch.log(alpha + EPS), dim=1)
		# Take mean of negative entropy across batch
		mean_neg_entropy = torch.mean(neg_entropy, dim=0)
		# KL loss of alpha with uniform categorical variable
		kl_loss = log_dim + mean_neg_entropy
		return kl_loss.mean()

	def kl(self, p, q):
		return torch.sum(p * torch.log((p + EPS) / (q + EPS)))

	def weighted_kl(self, weight, p, q):
		# disc_dim = int(p.size()[-1])
		# log_dim = torch.Tensor([np.log(disc_dim)]).to(self.device)
		neg_entropy = torch.sum(p * torch.log((p + EPS) / (q + EPS)), dim=1)
		mean_neg_entropy = torch.mean( weight.view(neg_entropy.shape) * neg_entropy, dim=0)
		return mean_neg_entropy.mean()

	def weighted_mse_loss(self, weight, input, output):
		return torch.mean(weight * (input - output) ** 2)

	def weighted_kl_discrete_loss(self, weight, alpha):
		disc_dim = int(alpha.size()[-1])
		log_dim = torch.Tensor([np.log(disc_dim)]).to(self.device)
		# Calculate negative entropy of each row
		neg_entropy = torch.sum(alpha * torch.log(alpha + EPS), dim=1)
		# Take mean of negative entropy across batch
		mean_neg_entropy = torch.mean(weight.view(neg_entropy.shape) * (neg_entropy + log_dim), dim=0)
		# KL loss of alpha with uniform categorical variable
		kl_loss = mean_neg_entropy
		# print('weight.shape', weight.shape)
		# print('kl_loss.shape', kl_loss.shape)
		# print('mean_neg_entropy.shape', mean_neg_entropy.shape)
		return kl_loss.mean()

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		# torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

		torch.save(self.posterior.state_dict(), filename + "_posterior")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))

		# self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		# self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.posterior.load_state_dict(torch.load(filename + "_posterior"))