import os
from unityagents import UnityEnvironment
import numpy as np
from time import time

train_mode = True
max_steps = 1000

if os.name == 'nt':  # windows
	binary = os.path.join('cicuit2', 'circuit_2')
else:
	binary = 'circuit_linux/circuit_linux.x86_64'

env = UnityEnvironment(file_name=binary, worker_id=0)

print(str(env))

default_brain = env.brain_names[0]
brain = env.brains[default_brain]
config = {"WrongDirectionPenalty" : 0.01, 'PenaltyCarCollision': 1.0, 'MaxAngleReward': 35,'TimePenalty' : 0.015}

env_info = env.reset(train_mode=train_mode, config=config)[default_brain]

# print('env_info', vars(env_info))

s = env_info.states
ob = env_info.observations
nb_agents = np.shape(s if len(s) > len(ob) else ob)[0]
print('state', np.shape(s))
print('observation', np.shape(ob))
print('nb_agents', nb_agents)


t_init = time()
global_step = 0

resets = 0

while True:
    s = env_info.states
    ob = env_info.observations

    a = [0]

    env_info = env.step(a)[default_brain]
    reward = env_info.rewards
    # print('reward', np.shape(reward))
    done = env_info.local_done
    # print('done', np.shape(done))
    resets += np.sum(done)

    global_step += 1

    if global_step > max_steps:
        break

fps = global_step / (time() - t_init)
print('steps per second {:.4f}'.format(fps))
print('av. episode length', global_step * nb_agents / (resets + 1))

# on bart7: steps per second 49.8968
# on cpu: steps per second 6.3067