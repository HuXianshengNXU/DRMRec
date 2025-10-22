
from torch.nn import MSELoss
import torch

class MADR:

	def __init__(self, policy_net, lr=3e-4, gamma=0.99, eps_clip=0.2):
		self.policy = policy_net
		self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
		self.gamma = gamma
		self.eps_clip = eps_clip
		self.mse_loss = MSELoss()

	def update(self, states, actions, old_log_probs, rewards, values, timesteps, epochs=5):
		if isinstance(rewards, list):
			rewards = torch.tensor(rewards).cuda()

		if len(rewards.shape) == 1:
			rewards = rewards.repeat_interleave(states.shape[0] // len(rewards))

		discounted_rewards = []
		cum_reward = 0
		for r in reversed(rewards):
			cum_reward = r + self.gamma * cum_reward
			discounted_rewards.insert(0, cum_reward)
		discounted_rewards = torch.tensor(discounted_rewards).cuda()
		discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)

		if values.shape != discounted_rewards.shape:
			values = values.repeat(discounted_rewards.shape[0] // values.shape[0])

		advantages = discounted_rewards - values.detach()

		for _ in range(epochs):
			log_probs = self.policy.get_log_prob(states, timesteps, actions)
			ratio = torch.exp(log_probs - old_log_probs.detach())
			surr1 = ratio * advantages
			surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
			loss = -torch.min(surr1, surr2).mean() + 0.5 * self.mse_loss(values, discounted_rewards)

			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()