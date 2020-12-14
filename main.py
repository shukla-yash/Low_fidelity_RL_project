#! /usr/bin/env python3

import gym
import numpy as np

import rospy
from discretized_movement.msg import worldstate
from SimpleDQN import SimpleDQN

# import gym_novel_gridworlds
# import sys
# sys.path.append('gym_novel_gridworlds/envs')
# from novel_gridworld_v0_env import NovelGridworldV0Env
# import matplotlib.pyplot as plt

CURRENT_WORLDSTATE: worldstate
CURRENT_WORLDSTATE = worldstate()
CURRENT_WORLDSTATE_LAST_SET: float
CURRENT_WORLDSTATE_LAST_SET = 0


def world_state_callback(msg: worldstate):
    global CURRENT_WORLDSTATE
    CURRENT_WORLDSTATE = msg
    global CURRENT_WORLDSTATE_LAST_SET
    CURRENT_WORLDSTATE_LAST_SET = rospy.get_time()


def get_loc(obj: str):
    if obj == 'agent':
        return [CURRENT_WORLDSTATE.robot_state.x,
                CURRENT_WORLDSTATE.robot_state.y]

    if CURRENT_WORLDSTATE_LAST_SET == 0 or CURRENT_WORLDSTATE is None:
        return [0, 0]
    for n in range(0, CURRENT_WORLDSTATE.observed_objects.__len__()):
        if CURRENT_WORLDSTATE.observed_objects[n].name == obj:
            return [CURRENT_WORLDSTATE.observed_objects[n].x,
                    CURRENT_WORLDSTATE.observed_objects[n].y]

    return [0, 0]


if __name__ == "__main__":
    rospy.init_node("low_fidelity_rl_agent", anonymous=False)

    no_of_environmets = 1

    # Width of the grid (grid is 10,10. 1 Extra row and height for defining the boundary.)
    width = 11
    height = 11  # Height of the grid
    no_cube1 = 1  # No of cube1 in the env
    no_cube2 = 1  # No of cube2 in the env
    crafting_table = 1  # No of crafting tables in the env
    type_of_env = 2  # The goal task. Changing this param to 0 or 1 would make the goal intermediate tasks of the curriculum

    rate = rospy.Rate(.5)
    while CURRENT_WORLDSTATE is None:
        rate.sleep()
        if CURRENT_WORLDSTATE is None:
            rospy.logwarn(
                2, "Still haven't seen any worldstate information, will keep waiting...")

    cube1_loc = get_loc('cube1')
    cube2_loc = get_loc('cube2')
    crafting_table_loc = get_loc('crafting_table')
    agent_loc = get_loc('agent')  # gripper location

    total_timesteps_array = []
    total_reward_array = []
    avg_reward_array = []

    actionCnt = 6
    D = 33  # 8 beams x 4 items lidar + 1 inventory items
    NUM_HIDDEN = 10
    GAMMA = 0.95
    LEARNING_RATE = 1e-3
    DECAY_RATE = 0.99
    MAX_EPSILON = 0.1
    random_seed = 1

    final_status = True

    agent = SimpleDQN(actionCnt, D, NUM_HIDDEN, LEARNING_RATE,
                      GAMMA, DECAY_RATE, MAX_EPSILON, random_seed)
    agent.set_explore_epsilon(MAX_EPSILON)
    agent.load_model(0, 0, 3)
    agent.reset()
    print("loaded model")

    final_status = True

    env_id = 'NovelGridworld-v0'
    env = gym.make(env_id, map_width=width, map_height=height, items_quantity={'cube1': no_cube1, 'cube2': no_cube2, 'crafting_table': crafting_table},
                   items_loc={'cube1': cube1_loc, 'cube2': cube2_loc, 'crafting_table': crafting_table_loc}, agent_loc=agent_loc, goal_env=type_of_env, is_final=final_status)

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
        a = agent.process_step(obs, True)

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

            print("\n\nfinished episode = "+str(episode) +
                  " with " + str(reward_sum)+"\n")

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
