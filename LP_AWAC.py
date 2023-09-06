import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class Actor(nn.Module):
    def __init__(self, state_dim, latent_dim, max_action):
        super(Actor, self).__init__()
        hidden_size = (256, 256)

        self.pi1 = nn.Linear(state_dim, hidden_size[0])
        self.pi2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.pi3 = nn.Linear(hidden_size[1], latent_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.pi1(state))
        a = F.relu(self.pi2(a))
        a = self.pi3(a)
        a = self.max_action * torch.tanh(a)

        return a

class ActorVAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action):
        super(ActorVAE, self).__init__()
        hidden_size = (256, 256)

        self.e1 = nn.Linear(state_dim + action_dim, hidden_size[0])
        self.e2 = nn.Linear(hidden_size[0], hidden_size[1])

        self.mean = nn.Linear(hidden_size[1], latent_dim)
        self.log_var = nn.Linear(hidden_size[1], latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, hidden_size[0])
        self.d2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.d3 = nn.Linear(hidden_size[1], action_dim)

        self.max_action = max_action
        self.action_dim = action_dim
        self.latent_dim = latent_dim

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        log_var = self.log_var(z)
        std = torch.exp(log_var/2)
        z = mean + std * torch.randn_like(std)

        u = self.decode(state, z)

        return u, z, mean, log_var

    def decode(self, state, z=None, clip=None):
        # When sampling from the VAE, the latent vector is clipped
        if z is None:
            clip = self.max_action
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-clip, clip)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        a = self.d3(a)
        return a

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        hidden_size = (256, 256)

        self.l1 = nn.Linear(state_dim + action_dim, hidden_size[0])
        self.l2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.l3 = nn.Linear(hidden_size[1], 1)

        self.l4 = nn.Linear(state_dim + action_dim, hidden_size[0])
        self.l5 = nn.Linear(hidden_size[0], hidden_size[1])
        self.l6 = nn.Linear(hidden_size[1], 1)

        self.v1 = nn.Linear(state_dim, hidden_size[0])
        self.v2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.v3 = nn.Linear(hidden_size[1], 1)

    def forward(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = (self.l3(q1))

        q2 = F.relu(self.l4(torch.cat([state, action], 1)))
        q2 = F.relu(self.l5(q2))
        q2 = (self.l6(q2))
        return q1, q2

    def q1(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = (self.l3(q1))
        return q1

    def v(self, state):
        v = F.relu(self.v1(state))
        v = F.relu(self.v2(v))
        v = (self.v3(v))
        return v

class LP_AWAC(object):
    def __init__(self, state_dim, action_dim, latent_dim, max_action,
                 device_id=-1,
                 discount=0.99, tau=0.005, vae_lr=3e-4, actor_lr=3e-4, critic_lr=3e-4,
                 max_latent_action=1, kl_beta=1.0, scale=5.0,
                 no_noise=True, doubleq_min=1.0):

        if device_id == -1:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu")

        self.actor_vae = ActorVAE(state_dim, action_dim, latent_dim, max_latent_action).to(self.device)
        self.actor_vae_target = copy.deepcopy(self.actor_vae)
        self.actorvae_optimizer = torch.optim.Adam(self.actor_vae.parameters(), lr=vae_lr)

        self.actor = Actor(state_dim, latent_dim, max_latent_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.latent_dim = latent_dim
        self.max_action = max_action
        self.max_latent_action = max_latent_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.tau_vae = tau
        self.scale=scale

        self.kl_beta = kl_beta
        self.no_noise = no_noise
        self.doubleq_min = doubleq_min
        self.min_v = None
        self.max_v = None

        self.total_it = 0



    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            latent_a = self.actor(state)
            action = self.actor_vae_target.decode(state, z=latent_a).cpu().data.numpy().flatten()

        return action

    def kl_loss(self, mu, log_var):
        KL_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1).view(-1, 1)
        return KL_loss

    def get_target_q(self, state, actor_net, critic_net, use_noise=False):
        latent_action = actor_net(state)
        if use_noise:
            latent_action += (torch.randn_like(latent_action) * 0.1).clamp(-0.2, 0.2)
        actor_action = self.actor_vae_target.decode(state, z=latent_action)
        
        target_q1, target_q2 = critic_net(state, actor_action)
        target_q = torch.min(target_q1, target_q2)*self.doubleq_min + torch.max(target_q1, target_q2)*(1-self.doubleq_min)

        return target_q

    def train(self, replay_buffer, batch_size=100, envname=''):
        self.total_it += 1

        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        if 'antmaze' in envname:
            reward = reward - 1

        # Critic Training
        with torch.no_grad():
            next_target_v = self.critic.v(next_state)
            target_Q = reward + not_done * self.discount * next_target_v
            target_v = self.get_target_q(state, self.actor_target, self.critic_target, use_noise=True)

        current_Q1, current_Q2 = self.critic(state, action)
        current_v = self.critic.v(state)

        v_loss = F.mse_loss(current_v, target_v)

        critic_loss_1 = F.mse_loss(current_Q1, target_Q)
        critic_loss_2 = F.mse_loss(current_Q2, target_Q)
        critic_loss = critic_loss_1 + critic_loss_2 + v_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # compute adv and weight
        current_v = self.critic.v(state)

        next_q = self.get_target_q(next_state, self.actor_target, self.critic_target)
        q_action = reward + not_done * self.discount * next_q
        adv = (q_action - current_v)

        width = torch.max(adv).detach() - torch.min(adv).detach()
        weights = torch.exp(self.scale * (adv - adv.max()) / width)
        weights = weights / weights.mean()

        # train weighted CVAE
        recons_action, z_sample, mu, log_var = self.actor_vae(state, action)

        recons_loss_ori = F.mse_loss(recons_action, action, reduction='none')
        recon_loss = torch.sum(recons_loss_ori, 1).view(-1, 1)
        KL_loss = self.kl_loss(mu, log_var)
        actor_vae_loss = (recon_loss + KL_loss*self.kl_beta)*weights.detach()

        actor_vae_loss = actor_vae_loss.mean()
        self.actorvae_optimizer.zero_grad()
        actor_vae_loss.backward()
        self.actorvae_optimizer.step()

        # train latent policy
        latent_actor_action = self.actor(state)
        actor_action = self.actor_vae_target.decode(state, z=latent_actor_action)
        q_pi = self.critic.q1(state, actor_action)

        actor_loss = -q_pi.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update Target Networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor_vae.parameters(), self.actor_vae_target.parameters()):
            target_param.data.copy_(self.tau_vae * param.data + (1 - self.tau_vae) * target_param.data)

        assert (np.abs(np.mean(target_Q.cpu().data.numpy())) < 1e6)

        if self.total_it % 1000 == 0:
            print(self.total_it, ": critic loss", (critic_loss_1+critic_loss_2).item())

        return (critic_loss_1+critic_loss_2).item()

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_vae.state_dict(), filename + "_actor_vae")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_vae.load_state_dict(torch.load(filename + "_actor_vae"))
