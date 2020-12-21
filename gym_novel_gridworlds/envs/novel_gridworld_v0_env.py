# Author: Yash Shukla
# Email: yash.shukla@tufts.edu

import math

import actionlib
import discretized_movement.msg

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import get_cmap
import time
import gym
from gym import error, spaces, utils
from gym.utils import seeding


class NovelGridworldV0Env(gym.Env):

    def __init__(self, map_width=None, map_height=None, items_id=None, items_quantity=None, items_loc = None, agent_loc = None, goal_env = None, is_final = False):
        # NovelGridworldV7Env attributes
        self.env_name = 'NovelGridworld-v0'
        self.map_width = 11
        self.map_height = 11
        self.map = np.zeros((self.map_width, self.map_height), dtype=int)  # 2D Map
        self.agent_location = (1, 1)  # row, column
        # self.direction_id = {'NORTH': 0, 'SOUTH': 1, 'WEST': 2, 'EAST': 3}
        self.direction_id = {'NORTH': 0}
        self.agent_facing_str = 'NORTH'
        self.agent_facing_id = self.direction_id[self.agent_facing_str]
        self.block_in_front_str = 'air'
        self.block_in_front_id = 0  # air
        self.block_in_front_location = (0, 0)  # row, column
        self.items = ['wall', 'cube1', 'cube2', 'crafting_table']
        self.items_id = self.set_items_id(self.items)  # {'crafting_table': 1, 'pogo_stick': 2, ...}  # air's ID is 0
        # items_quantity when the episode starts, do not include wall, quantity must be more than 0
        self.items_quantity = {'cube1': 1, 'cube2': 1, 'crafting_table': 1}
        self.items_loc = {'cube1' : [4,6] , 'cube2': [7,9], 'crafting_table': [3,3]}
        self.agent_loc = [5,5]

        # Action Space
        self.action_str = {0: 'Forward', 1: 'Left', 2: 'Right', 3: 'Backward', 4: 'PickUp', 5: 'Drop'}
        self.goal_env = 2
        self.action_space = spaces.Discrete(len(self.action_str))
        self.recipes = {}
        self.last_action = 0  # last actions executed
        self.step_count = 0  # no. of steps taken

        # Observation Space
        self.num_beams = 8
        self.max_beam_range = 40
        self.items_lidar = ['wall', 'crafting_table', 'cube1', 'cube2']
        self.items_id_lidar = self.set_items_id(self.items_lidar)
        self.low = np.array([0] * (len(self.items_lidar) * self.num_beams) + [0] * 1)
        self.high = np.array([self.max_beam_range] * (len(self.items_lidar) * self.num_beams) + [2] * 1)  # maximum 5 trees present in the environment
        self.observation_space = spaces.Box(self.low, self.high, dtype=int)

        # Reward
        self.last_reward = 0  # last received reward
        self.last_done = False  # last done
        self.reward_done = 1000
        self.reward_break = 200
        self.episode_timesteps = 250

        if map_width is not None:
            self.map_width = map_width
        if map_height is not None:
            self.map_height = map_height
        if items_id is not None:
            self.items_id = items_id
        if items_quantity is not None:
            self.items_quantity = items_quantity
        if items_loc is not None:
            self.items_loc = items_loc
        if agent_loc is not None:
            self.agent_loc = agent_loc
        if goal_env is not None:
            self.goal_env = goal_env
        if is_final == True:
            self.reward_break = 200

        self.current_pickup_state = 0
        self.current_pickup_item = 0 # 0 for cube1, 1 for cube2
        self.dropped_items = 0
        self.target_dropped_items = self.items_quantity['cube1'] + self.items_quantity['cube2']

        self.kinematics_client = actionlib.SimpleActionClient('simplified_kinematics', discretized_movement.msg.MoveAction)
        self.kinematics_client.wait_for_server()
        self.interation_client = actionlib.SimpleActionClient('simplified_interaction', discretized_movement.msg.InteractAction)
        self.interation_client.wait_for_server()



    def reset(self):

        # Variables to reset for each reset:
        self.available_locations = []
        self.not_available_locations = []
        self.last_action = 0  # last actions executed
        self.step_count = 0  # no. of steps taken
        self.last_reward = 0  # last received reward
        self.last_done = False  # last done

        self.current_pickup_state = 0
        self.current_pickup_item = 0 # 0 for cube1, 1 for cube2
        self.dropped_items = 0
        self.target_dropped_items = self.items_quantity['cube1'] + self.items_quantity['cube2']

        self.map = np.zeros((self.map_width - 2, self.map_height - 2), dtype=int)  # air=0
        self.map = np.pad(self.map, pad_width=1, mode='constant', constant_values=self.items_id['wall'])

        """
        available_locations: locations 1 block away from the wall are valid locations to place items and agent
        available_locations: locations that do not have item placed
        """
        for r in range(2, self.map_width - 2):
            for c in range(2, self.map_height - 2):
                self.available_locations.append((r, c))

        r_cube1 = int(self.items_loc['cube1'][0])
        c_cube1 = int(self.items_loc['cube1'][1])
        self.map[r_cube1][c_cube1] = self.items_id['cube1']

        r_cube2 = int(self.items_loc['cube2'][0])
        c_cube2 = int(self.items_loc['cube2'][1])
        self.map[r_cube2][c_cube2] = self.items_id['cube2']

        r_crafting_table = int(self.items_loc['crafting_table'][0])
        c_crafting_table = int(self.items_loc['crafting_table'][1])
        self.map[r_crafting_table][c_crafting_table] = self.items_id['crafting_table']

        r_agent = int(self.agent_loc[0])
        c_agent = int(self.agent_loc[1])
        self.agent_location = (r_agent, c_agent)

        # Agent facing direction
        self.set_agent_facing(direction_str=np.random.choice(list(self.direction_id.keys()), size=1)[0])

        # Update after each reset
        observation = self.get_observation()
        self.update_block_in_front()

        return observation

    def get_lidarSignal(self):
        """
        Send several beans (self.num_beams) at equally spaced angles in 360 degrees in front of agent within a range
        For each bean store distance (beam_range) for each item in items_id_lidar if item is found otherwise 0
        and return lidar_signals
        """

        direction_radian = {'NORTH': np.pi, 'SOUTH': 0, 'WEST': 3 * np.pi / 2, 'EAST': np.pi / 2}

        # Shoot beams in 360 degrees in front of agent
        angles_list = np.linspace(direction_radian[self.agent_facing_str] - np.pi,
                                  direction_radian[self.agent_facing_str] + np.pi,
                                  self.num_beams + 1)[:-1]  # 0 and 360 degree is same, so removing 360

        lidar_signals = []
        r, c = self.agent_location
        for angle in angles_list:
            x_ratio, y_ratio = np.round(np.cos(angle), 2), np.round((np.sin(angle)), 2)
            beam_signal = np.zeros(len(self.items_id_lidar), dtype=int)#

            # Keep sending longer beams until hit an object or wall
            for beam_range in range(1, self.max_beam_range+1):
                r_obj = r + np.round(beam_range * x_ratio)
                c_obj = c + np.round(beam_range * y_ratio)
                obj_id_rc = self.map[int(r_obj)][int(c_obj)]

                # If bean hit an object or wall
                if obj_id_rc != 0:
                    item = list(self.items_id.keys())[list(self.items_id.values()).index(obj_id_rc)]
                    if item in self.items_id_lidar:
                        obj_id_rc = self.items_id_lidar[item]
                        beam_signal[obj_id_rc - 1] = beam_range
                    break

            lidar_signals.extend(beam_signal)

        return lidar_signals

    def set_agent_facing(self, direction_str):

        self.agent_facing_str = direction_str
        self.agent_facing_id = self.direction_id[self.agent_facing_str]

        '''
        self.agent_facing_str = list(self.direction_id.keys())[list(self.direction_id.values()).index(self.agent_facing_id)]
        '''

    def set_items_id(self, items):

        items_id = {}
        for item in sorted(items):
            items_id[item] = len(items_id) + 1

        return items_id

    def get_observation(self):
        """
        observation is lidarSignal + inventory_items_quantity
        :return: observation
        """

        lidar_signals = self.get_lidarSignal()
        # observation = lidar_signals + [self.inventory_items_quantity[item] for item in
        #                                sorted(self.inventory_items_quantity)]

        observation = lidar_signals + [self.current_pickup_item]

        # print(observation)
        # time.sleep(5.0)2
        return np.array(observation)

    def interact(self):
        ''' Rather than just end the episode, we try to "interact" with the target in some way.  '''
        interact_goal = discretized_movement.msg.InteractGoal()
        interact_goal.action.interact = interact_goal.action.GRAB
        self.interation_client.send_goal_and_wait(interact_goal)
        result = self.interation_client.get_result()

    def step(self, action):
        """
        Actions: {0: 'Forward', 1: 'Left', 2: 'Right', 3: 'Break'}
        """
        move_goal_set = False
        move_goal = discretized_movement.msg.MoveGoal()
        interact_goal_set = False
        interact_goal = discretized_movement.msg.InteractGoal()

        self.last_action = action
        r, c = self.agent_location

        done = False
        reward = -1  # default reward
        # Forward
        if action == 0:
            if self.agent_facing_str == 'NORTH' and self.map[r - 1][c] == 0:
                self.agent_location = (r - 1, c)
                move_goal_set = True
                move_goal.move.direction = move_goal.move.UP
        # Left
        elif action == 1:
            if self.agent_facing_str == 'NORTH' and self.map[r][c-1] == 0:
                self.agent_location = (r, c-1)
                move_goal_set = True
                move_goal.move.direction = move_goal.move.LEFT

        # Right
        elif action == 2:
            if self.agent_facing_str == 'NORTH' and self.map[r][c+1] == 0:
                self.agent_location = (r, c+1)
                move_goal_set = True
                move_goal.move.direction = move_goal.move.RIGHT

        # Backward
        elif action == 3:
            if self.agent_facing_str == 'NORTH' and self.map[r+1][c] == 0:
                self.agent_location = (r+1, c)
                move_goal_set = True
                move_goal.move.direction = move_goal.move.DOWN


        # PickUp
        elif action == 4:
            self.update_block_in_front()
            # If block in front is not air and wall, place the block in front in inventory
            interact_goal.action.interact = interact_goal.action.GRAB
            interact_goal_set = True

            if self.block_in_front_str == 'cube1' or self.block_in_front_str == 'cube2':
                block_r, block_c = self.block_in_front_location
                self.map[block_r][block_c] = 0
                reward = self.reward_break
                self.current_pickup_state = 1
                if self.block_in_front_str =='cube1':
                    self.current_pickup_item = 1
                elif self.block_in_front_str == 'cube2':
                    self.current_pickup_item = 2

        # Release
        elif action == 5:
            self.update_block_in_front()
            interact_goal.action.interact = interact_goal.action.RELEASE
            interact_goal_set = True

            if self.block_in_front_str == 'crafting_table':
                if self.current_pickup_state == 1:
                    reward = self.reward_break
                    self.dropped_items += 1
                    self.current_pickup_state = 0
                    self.current_pickup_item = 0

        # This isn't accessed since it seems like it's redundant information
        # (you're already tracking it in the gridworld), but you can get what the
        # robot perceives the world to be this way. Useful if it ever fails to
        # grab, release, or move properly. You can get the worldstate or success true/fail from it.
        result = None
        if move_goal_set:
            self.kinematics_client.send_goal_and_wait(move_goal)
            result = self.kinematics_client.get_result()
        if interact_goal_set:
            # The agent needs to be in front of an object on the same plane, but the robot needs to be on the object
            # on a different plane... So we go forward, then request the interaction, then return to where we were a
            # moment earlier.
            move_goal.move.direction = move_goal.move.UP
            self.kinematics_client.send_goal_and_wait(move_goal)

            self.interation_client.send_goal_and_wait(interact_goal)
            result = self.interation_client.get_result()

            move_goal.move.direction = move_goal.move.DOWN
            self.kinematics_client.send_goal_and_wait(move_goal)

        # Update after each step
        observation = self.get_observation()
        self.update_block_in_front()
        if self.goal_env == 0: # If the goal is navigation
            if not self.block_in_front_id == 0 and not self.block_in_front_str == 'wall':
                done = True
                reward = self.reward_done

        if self.goal_env == 1: # If the goal is pickup
            if self.current_pickup_item > 0:
                reward = self.reward_done
                done = True

        if self.goal_env == 2:
            if self.dropped_items == self.target_dropped_items:
                reward = self.reward_done
                done = True

        info = {}

        # Update after each step
        self.step_count += 1
        self.last_reward = reward
        self.last_done = done

        # if done == False and self.step_count == self.episode_timesteps:
        #     done = True

        return observation, reward, done, info

    def update_block_in_front(self):
        r, c = self.agent_location


        if self.agent_facing_str == 'NORTH':
            print(self.map)
            self.block_in_front_id = self.map[r - 1][c]
            self.block_in_front_location = (r - 1, c)


        if self.block_in_front_id == 0:
            self.block_in_front_str = 'air'
        else:
            self.block_in_front_str = list(self.items_id.keys())[
                list(self.items_id.values()).index(self.block_in_front_id)]

    def render(self, mode='human', title=None):

        color_map = "gist_ncar"

        if title is None:
            title = self.env_name

        r, c = self.agent_location
        x2, y2 = 0, 0
        if self.agent_facing_str == 'NORTH':
            x2, y2 = 0, -0.01
        elif self.agent_facing_str == 'SOUTH':
            x2, y2 = 0, 0.01
        elif self.agent_facing_str == 'WEST':
            x2, y2 = -0.01, 0
        elif self.agent_facing_str == 'EAST':
            x2, y2 = 0.01, 0

        plt.figure(title, figsize=(9, 5))
        plt.imshow(self.map, cmap=color_map, vmin=0, vmax=len(self.items_id))
        plt.arrow(c, r, x2, y2, head_width=0.7, head_length=0.7, color='white')
        plt.title('NORTH', fontsize=10)
        plt.xlabel('SOUTH')
        plt.ylabel('WEST')
        plt.text(self.map_width, self.map_width // 2, 'EAST', rotation=90)
        # plt.text(self.map_size, self.map_size // 2, 'EAST', rotation=90)
        # plt.colorbar()
        # plt.grid()

        info = '\n'.join(["               Info:             ",
                          "Env: "+self.env_name,
                          "Steps: " + str(self.step_count),
                          "Agent Facing: " + self.agent_facing_str,
                          "Action: " + self.action_str[self.last_action],
                          "Reward: " + str(self.last_reward),
                          "Done: " + str(self.last_done)])
        props = dict(boxstyle='round', facecolor='w', alpha=0.2)
        plt.text(-(self.map_width // 2) - 0.5, 2.25, info, fontsize=10, bbox=props)  # x, y

        # plt.text(-(self.map_size // 2) - 0.5, 2.25, info, fontsize=10, bbox=props)  # x, y

        if self.last_done:
            you_win = "YOU WIN "+self.env_name+"!!!"
            props = dict(boxstyle='round', facecolor='w', alpha=1)
            # plt.text(0 - 0.1, (self.map_size // 2), you_win, fontsize=18, bbox=props)
            plt.text(0 - 0.1, (self.map_width // 2), you_win, fontsize=18, bbox=props)

        cmap = get_cmap(color_map)

        legend_elements = [Line2D([0], [0], marker="^", color='w', label='agent', markerfacecolor='w', markersize=12,
                                  markeredgewidth=2, markeredgecolor='k'),
                           Line2D([0], [0], color='w', label="INVENTORY:")]
        # for item in sorted(self.inventory_items_quantity):
        #     rgba = cmap(self.items_id[item] / len(self.items_id))
        #     legend_elements.append(Line2D([0], [0], marker="s", color='w',
        #                                   label=item + ': ' + str(self.inventory_items_quantity[item]),
        #                                   markerfacecolor=rgba, markersize=16))
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.55, 1.02))  # x, y

        plt.tight_layout()
        plt.pause(0.01)
        plt.clf()

    def close(self):
        return
