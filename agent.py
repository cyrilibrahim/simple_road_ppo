import torch
from torch import nn
from nnet import Model
from torch.distributions import Categorical


class PPOAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Model()

    def act(self, obs):
        policy_probs, value = self.model(obs)
        policy_dist = Categorical(policy_probs)
        action = policy_dist.sample()
        action_log_prob = policy_dist.log_prob(action)
        return action, action_log_prob, value

    def evaluate_action(self, obs, action):
        policy_probs, value = self.model(obs)
        policy_dist = Categorical(policy_probs)
        action_likelihood = policy_dist.log_prob(action.squeeze())
        action_likelihood = action_likelihood.view(-1, 1)
        dist_entropy = (policy_probs * policy_probs.log()).sum(1).view(-1, 1)
        return action_likelihood, dist_entropy, value

    def forward(self, obs):
        _, value = self.model(obs)
        return value
