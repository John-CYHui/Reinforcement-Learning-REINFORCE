import numpy as np
import gym
import csv
import os
import glob
import pickle as pk
from agent import ReinforceBaseLineAgent

# Create results folder and clear contents
if not os.path.exists('./results'):
    os.makedirs('./results')

files = glob.glob('./results/*')
for f in files:
    os.remove(f)

# Initialize environment
max_steps = 20000
gym.envs.register(
    id='MountainCarMyEasyVersion-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=20000,      # MountainCar-v0 uses 200
)
env = gym.make('MountainCarMyEasyVersion-v0')
min_position, min_velocity = env.observation_space.low
max_position, max_velocity = env.observation_space.high


# Initialize REINFORCE Baseline agent
step_size_theta = 2e-8
num_tilings = 16
step_size_w = 0.1 / num_tilings
agent_info = {'min_position': min_position, 'min_velocity':min_velocity,
            'max_position': max_position, 'max_velocity': max_velocity,
            'num_actions': env.action_space.n, 'iht_size': 4096,
            'num_tilings': num_tilings, 'num_tiles': 8,
            'step_size_theta': step_size_theta, 'step_size_w': step_size_w,
            'discount_factor': 1}

agent = ReinforceBaseLineAgent()
agent.agent_init(agent_info)


# Episodes rollout
num_episode = 10000

for episode in range(num_episode):
    print(f'Episode: {episode+1}')
    if (episode+1) % 100 == 0:
        with open(f'./results/agent_episode_{episode+1}.pk', 'wb') as f:
            pk.dump(agent, f)

    # Generate_episodes
    state_ls = []
    action_ls = []
    reward_ls = []
    state = env.reset()

    total_reward = 0
    terminal = False
    while not terminal:
        #env.render()
        action = agent.agent_step(state)
        state_ls.append(state)
        action_ls.append(action)
        state, reward, terminal, info = env.step(action)
        total_reward += reward
        if terminal:
            with open('./results/episode_rewards.csv', 'a', newline='') as f:
                write = csv.writer(f)
                write.writerow([episode+1, total_reward])
            
            
            break
        reward_ls.append(reward)

    if total_reward == -max_steps:
        print(agent.theta)
        pass
    else:
        for t in range(len(reward_ls)):
            # Naive implementation
            # g = 0
            # for k in range(t + 1, len(state_ls)):
            #     g = g + agent.discount_factor ** (k-t-1) * reward_ls[k-1]

            # state = state_ls[t]
            # action = action_ls[t]
            # agent.update_weight(action, state, g, t)

            # Faster implementation
            g = 0
            power_arr = np.arange(t+1, len(state_ls))
            agent_discount_factor_arr = np.array([agent.discount_factor] * len(power_arr))
            reward_arr = np.array(reward_ls[t:len(state_ls)])
            g = np.power(agent_discount_factor_arr, power_arr)
            g = np.dot(g, reward_arr)

            state = state_ls[t]
            action = action_ls[t]
            active_state_feature = agent.get_state_feature(state)
            delta = g - agent.get_state_value(active_state_feature)
            agent.update_w_weight(state, delta)
            agent.update_theta_weight(action, state, delta, t)

        print(agent.theta)
        print(agent.w)
env.close()
