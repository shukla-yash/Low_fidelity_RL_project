import os
import sys

import gym
import time
import numpy as np
import gym_novel_gridworlds

# sys.path.append('gym_novel_gridworlds/envs')
# from novel_gridworld_v0_env import NovelGridworldV0Env
from SimpleDQN import SimpleDQN
import matplotlib.pyplot as plt


if __name__ == "__main__":

	no_of_environmets = 1

	width = 11 # Width of the grid (grid is 10,10. 1 Extra row and height for defining the boundary.)
	height = 11 # Height of the grid
	no_cube1 = 1 # No of cube1 in the env
	no_cube2 = 1 # No of cube2 in the env
	crafting_table = 1 # No of crafting tables in the env
	type_of_env = 2 # The goal task. Changing this param to 0 or 1 would make the goal intermediate tasks of the curriculum

	cube1_loc = [4,7] # Location of cube1 on the grid
	cube2_loc = [7,9] # Location of cube2 on the grid
	crafting_table_loc = [3,3] # Location of crafting table on the grid
	agent_loc = [5,5] # Agent loc

	total_timesteps_array = []
	total_reward_array = []
	avg_reward_array = []

	actionCnt = 6
	D = 33 #8 beams x 4 items lidar + 1 inventory items
	NUM_HIDDEN = 10
	GAMMA = 0.95
	LEARNING_RATE = 1e-3
	DECAY_RATE = 0.99
	MAX_EPSILON = 0.1
	random_seed = 1

	final_status = True

	agent = SimpleDQN(actionCnt,D,NUM_HIDDEN,LEARNING_RATE,GAMMA,DECAY_RATE,MAX_EPSILON,random_seed)
	agent.set_explore_epsilon(MAX_EPSILON)
	agent.load_model(0,0,3)
	agent.reset()
	print("loaded model")

	final_status = True

	env_id = 'NovelGridworld-v0'
	env = gym.make(env_id, map_width = width, map_height = height, items_quantity = {'cube1': no_cube1, 'cube2': no_cube2, 'crafting_table': crafting_table}, \
		items_loc = {'cube1' : cube1_loc, 'cube2': cube2_loc, 'crafting_table': crafting_table_loc}, agent_loc = agent_loc, goal_env = type_of_env, is_final = final_status)
	
	t_step = 0
	episode = 0
	t_limit = 500
	reward_sum = 0
	reward_arr = []
	avg_reward = []
	done_arr = []
	env_flag = 0

	env.render()
	env.reset()

	while True:
		
		# get obseration from sensor
		env.render()
		obs = env.get_observation()
	
		# act 
		a = agent.process_step(obs,True)
		
		new_obs, reward, done, info = env.step(a)

		# give reward
		agent.give_reward(reward)
		reward_sum += reward
		
		t_step += 1
		
		if t_step > t_limit or done == True:
			
			# finish agent
			if done == True:
				done_arr.append(1)
			elif t_step > t_limit:
				done_arr.append(0)
			
			print("\n\nfinished episode = "+str(episode)+" with " +str(reward_sum)+"\n")

			reward_arr.append(reward_sum)
			avg_reward.append(np.mean(reward_arr[-40:]))

			total_reward_array.append(reward_sum)
			avg_reward_array.append(np.mean(reward_arr[-40:]))
			total_timesteps_array.append(t_step)
	
			done = True
			t_step = 0
			agent.finish_episode()
		
			# # update after every episode
			# if episode % 10 == 0:
			# 	agent.update_parameters()
		
			# reset environment
			episode += 1

			env.reset()
			env.render()
			reward_sum = 0
	
			# quit after some number of episodes
			if episode > 50:

				break
