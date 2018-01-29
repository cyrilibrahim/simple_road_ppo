from unityagents import UnityEnvironment
from tensorboardX import SummaryWriter
from agent import PPOAgent
from storage import RolloutStorage
from utils import img_to_tensor
import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from args import get_args
import tensorflow as tf
args = get_args()
torch.set_default_tensor_type('torch.cuda.FloatTensor')

writer = SummaryWriter(log_dir='logs/circuit')
env = UnityEnvironment(file_name='circuit2/circuit_2', worker_id=0)



print(str(env))
train_mode = True

agent = PPOAgent()
agent.model = agent.model.cuda()
load_weights = True
if(load_weights):
    agent.model.load_state_dict(torch.load("saved_weights/saved_model_ppo_epoch_23040"))

optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)

default_brain = env.brain_names[0]
brain = env.brains[default_brain]
config = {"WrongDirectionPenalty" : 0.01, 'PenaltyCarCollision': 1.0, 'MaxAngleReward': 35,'TimePenalty' : 0.015}


env_info = env.reset(train_mode=train_mode)[default_brain]
obs = env_info.observations[0]
obs = obs.squeeze(3)
obs_shape = obs.shape
action_shape = 1
rollouts = RolloutStorage(num_steps=args.num_steps, num_processes=1, obs_shape=obs_shape)
step = 0
episode = 0
ppo_update = 0
total_reward = 0

while True:  # nb episodes
    env_info = env.reset(train_mode=train_mode, config=config)[default_brain]
    obs = env_info.observations[0]
    obs = img_to_tensor(obs)
    while True:  # nb of steps
        action, action_log_prob, value = agent.act(obs)
        #action_cpu = action.data.numpy()
        action_cuda = action.data.cpu().numpy()
        #print(action_cuda)

        env_info = env.step(action_cuda)[default_brain]
        reward = torch.cuda.FloatTensor([env_info.rewards[0]])
        total_reward += env_info.rewards[0]

        mask = 0 if env_info.local_done[0] else 1
        mask = torch.cuda.FloatTensor([mask])
        rollouts.insert(step, obs.data, action.data, action_log_prob.data, value.data, reward, mask)
        step += 1
        obs = env_info.observations[0]
        obs = img_to_tensor(obs)

        if env_info.local_done[0]:
            if episode % 5 == 0:
                writer.add_scalar('episode_reward', total_reward, episode)
            episode += 1
            total_reward = 0

        if step == args.num_steps:
            step = 0
            break

        if env_info.local_done[0]:
            break

    if step == 0:
        print('ppo update')
        # do ppo update
        next_value = agent(obs)[0].data

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        for e in range(args.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(advantages,
                                                             args.num_mini_batch)

            for sample in data_generator:
                observations_batch, actions_batch, \
                return_batch, masks_batch, old_action_log_probs_batch, \
                adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                action_log_probs, dist_entropy, values = agent.evaluate_action(
                                                                    Variable(observations_batch),
                                                                    Variable(actions_batch))

                adv_targ = Variable(adv_targ)
                ratio = torch.exp(action_log_probs - Variable(old_action_log_probs_batch))
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()  # PPO's pessimistic surrogate (L^CLIP)

                value_loss = (Variable(return_batch) - values).pow(2).mean()

                optimizer.zero_grad()
                (value_loss + action_loss - dist_entropy.mean() * args.entropy_coef).backward()
                nn.utils.clip_grad_norm(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                ppo_update += 1

            if ppo_update // args.ppo_epoch % 5 == 0:
                writer.add_scalar('value_loss', value_loss.data.cpu().numpy(), ppo_update)
                writer.add_scalar('action_loss', action_loss.data.cpu().numpy(), ppo_update)
                writer.add_scalar('entropy_loss', dist_entropy.mean().data.cpu().numpy(), ppo_update)

                # Save model
                torch.save(agent.model.state_dict(), "saved_weights/saved_model_ppo_epoch_"+str(ppo_update))

    rollouts.after_update()

