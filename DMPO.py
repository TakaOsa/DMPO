import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class PosteriorApproximator(nn.Module):
	def __init__(self, state_dim, action_dim, latent_disc_dim, temperature=.67, hidden=(256, 256), device_id=-1):
		super(PosteriorApproximator, self).__init__()

		# Encoder architecture
		self.l1 = nn.Linear(state_dim + action_dim, hidden[0])
		self.l2 = nn.Linear(hidden[0], hidden[1])

		self.l3_disc = nn.Linear(hidden[1], latent_disc_dim)
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
			ind_max_alpha = torch.argmax(alpha, dim=1)
			one_hot_samples = F.one_hot(ind_max_alpha.view(-1, ), num_classes=alpha.shape[1])

			return one_hot_samples

	def forward(self, state, action):
		z_disc, alpha = self.encode(state, action)

		return z_disc, alpha

	def encode(self, state, action):
		sa = torch.cat([state, action], 1)
		h = F.relu(self.l1(sa))
		h = F.relu(self.l2(h))

		alpha = F.softmax(self.l3_disc(h))
		z_disc = self.sample_gumbel_softmax(alpha)

		return z_disc, alpha


def to_one_hot(y, class_num):
	one_hot = np.zeros((y.shape[0], class_num))
	one_hot[range(y.shape[0]), y] = 1.0

	return one_hot


class DMPO(object):
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
		scale=2.0,
		clip=0.5,
		beta=1.0,
		hidden=(256, 256),
		weight_type='',
		device_id=-1,
		lr=3e-4,
		info_reg = False,
		ilr=5e-7
	):

		print('torch.cuda.is_available()', torch.cuda.is_available())

		if device_id == -1:
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu")

		self.actor = Actor(state_dim, action_dim, latent_disc_dim, max_action, hidden=hidden).to(self.device)
		self.actor_target = copy.deepcopy(self.actor)
		self.posterior = PosteriorApproximator(state_dim, action_dim, latent_disc_dim, hidden=hidden, device_id=device_id).to(self.device)
		self.actor_optimizer = torch.optim.Adam(itertools.chain(self.posterior.parameters(), self.actor.parameters()),
												lr=lr)

		if info_reg:
			self.info_posterior = PosteriorApproximator(state_dim, action_dim, latent_disc_dim, hidden=hidden,
														device_id=device_id).to(self.device)
			self.info_optimizer = torch.optim.Adam(
				itertools.chain(self.actor.parameters(), self.info_posterior.parameters()),
				lr=ilr)

		self.critic = Critic(state_dim, action_dim, hidden=hidden).to(self.device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.latent_disc_dim = latent_disc_dim

		self.scale = scale
		self.weight_type = weight_type
		self.clip = clip
		self.beta = beta
		self.info_reg = info_reg

		self.action_dim = action_dim
		self.state_dim = state_dim

		self.total_it = 0

		print('DMPO')
		print('scale', self.scale)
		print('latent_disc_dim', self.latent_disc_dim)
		print('info_reg', self.info_reg)
		print('ilr', ilr)

	def select_action(self, state, return_z=False):
		self.posterior.eval()
		self.critic.eval()
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		latent, _ = self.select_latent(state, self.actor)
		self.critic.train()
		self.posterior.train()
		if return_z:
			return self.actor(state, latent).cpu().data.numpy().flatten(), latent.cpu().data.numpy().flatten()
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
		minQ = torch.min(Q1_pred, Q2_pred)

		ind = torch.argmax(torch.t(minQ.reshape(-1, state.shape[0])), dim=1)
		latent_set = latent_tile.view(eval_num, state.shape[0],self.latent_disc_dim).permute(1,0,2)
		latent_max = latent_set[torch.arange(state.size(0)), ind, :]

		return latent_max, torch.t(minQ.reshape(-1, state.shape[0]))[torch.arange(state.size(0)), ind].view(-1,1)

	def sample_latent(self, replay_buffer, batch_size=256, one_hot=True):
		# Sample replay buffer
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
		if one_hot:
			self.posterior.eval()
		else:
			self.posterior.train()
		latent_sample, alpha = self.posterior.encode(state, action)
		Q1, _ = self.critic(state, action)
		_, v = self.select_latent(state, self.actor)

		return latent_sample.cpu().data.numpy(), state.cpu().data.numpy(), action.cpu().data.numpy(), (Q1 - v).cpu().data.numpy()

	def eval_latent(self, state, nump=True):
		if nump:
			state = torch.FloatTensor(state.reshape(-1, self.state_dim)).to(self.device)

		self.posterior.eval()
		self.critic.eval()

		eval_num = self.latent_disc_dim
		z_int = np.arange(0, self.latent_disc_dim)
		latent_sample = to_one_hot(z_int, self.latent_disc_dim)

		latent_sample = torch.FloatTensor(latent_sample).to(self.device)
		latent_tile = (latent_sample.repeat(1, state.shape[0])).view(-1, self.latent_disc_dim)
		state_tile = state.repeat(eval_num, 1)

		action_tile = self.actor(state_tile, latent_tile)
		Q1_pred, Q2_pred = self.critic(state_tile, action_tile)
		minQ = torch.min(Q1_pred, Q2_pred)
		if nump:
			return torch.t(minQ.reshape(-1, state.shape[0])).cpu().data.numpy()
		else:
			return torch.t(minQ.reshape(-1, state.shape[0]))

	def train(self, replay_buffer, batch_size=256, envname=''):
		self.total_it += 1
		self.posterior.train()
		self.critic.train()
		critic_loss = None

		# Sample replay buffer
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		if 'antmaze' in envname:
			reward = reward - 1

		with torch.no_grad():
			next_latent_max, _ = self.select_latent(next_state, self.actor_target)
			# Select action according to policy and add clipped noise
			noise = (
					torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)

			next_action = (
					self.actor_target(next_state, next_latent_max) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)
		critic_loss = F.mse_loss(current_Q1, target_Q.detach()) + F.mse_loss(current_Q2, target_Q.detach())

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		if self.total_it % 1000 == 0:
			print(self.total_it, ": critic loss", critic_loss.item())

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:
			with torch.no_grad():
				sa = torch.cat([state, action], 1)
				Q1, Q2 = self.critic(state, action)
				Qmin = torch.min(Q1, Q2)

				latent, min_v = self.select_latent(state, self.actor_target)
				adv = Qmin - min_v

			weight = None
			if self.scale == 0.0:
				weight = torch.ones(size=adv.shape).to(self.device)
			else:
				if self.weight_type == 'clamp':
					weight = torch.exp(self.scale * adv).clamp(0, 100)
				else:
					width = torch.max(adv).detach() - torch.min(adv).detach()
					weight = torch.exp(self.scale * (adv - adv.max()) / width)
					weight = weight / weight.mean()

			# Compute mutual information loss
			if self.info_reg:
				label_latent = torch.randint(low=0, high=self.latent_disc_dim - 1,
											 size=(state.shape[0],)).to(self.device)
				latent_disc_sample = F.one_hot(label_latent,
											   num_classes=self.latent_disc_dim).type(torch.FloatTensor).to(self.device)

				action_pred_info = self.actor(state, latent_disc_sample.detach())
				_, alpha_info = self.info_posterior(state, action_pred_info)

				cross_entropy_loss = nn.CrossEntropyLoss()
				info_loss = cross_entropy_loss(alpha_info, label_latent.detach())

				self.info_optimizer.zero_grad()
				info_loss.backward()
				self.info_optimizer.step()

			# Compute actor loss
			latent_pred, alpha = self.posterior(state, action)
			action_pred = self.actor(state, latent_pred)

			q_latent = self.eval_latent(state, nump=False)
			q_zs = F.softmax(q_latent)

			kl_loss = self.weighted_kl(weight.detach(), alpha, q_zs.detach())
			actor_loss = self.weighted_mse_loss(weight.detach(), action_pred,
												action.detach()) * self.action_dim + self.beta * kl_loss

			# Optimize the actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		return critic_loss.item()

	def kl_discrete_loss(self, alpha):
		"""
		Calculates the KL divergence between a categorical distribution and a
		uniform categorical distribution.
		adopted from https://github.com/Schlumberger/joint-vae/
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
		neg_entropy = torch.sum(p * torch.log((p + EPS) / (q + EPS)), dim=1)
		mean_neg_entropy = torch.mean(weight.view(neg_entropy.shape) * neg_entropy, dim=0)
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
		# KL loss of alpha witind_max_alphah uniform categorical variable
		kl_loss = mean_neg_entropy
		return kl_loss.mean()

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.posterior.state_dict(), filename + "_posterior")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.posterior.load_state_dict(torch.load(filename + "_posterior"))