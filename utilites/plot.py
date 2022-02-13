import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./results - 20220211 tilings8_tiles_8_step_size_two_e_negative_8/episode_rewards.csv', names = ['episode', 'total reward'])
df_2 = pd.read_csv('./results - 20220211 tilings16_tiles_8_step_size_two_e_negative_8/episode_rewards.csv', names = ['episode', 'total reward'])
#df_3 = pd.read_csv('./results/episode_rewards.csv', names = ['episode', 'total reward'])

plt.plot(df['episode'], df['total reward'])
plt.plot(df_2['episode'], df_2['total reward'])
#plt.plot(df_3['episode'], df_3['total reward'])

plt.ylim([-5000, 0])
#plt.xlim([0, 4000])

plt.xlabel('Episode')
plt.ylabel('Total reward on episode')

plt.legend(['Tilings = 8, Tiles = 8, step size = 2e-8', 'Tilings = 16, Tiles = 8, step size = 2e-8'])
plt.savefig('./readme_pic/Total_reward_func_approx.jpg', dpi=200)