#!/usr/bin/env python
# coding: utf-8

# In[1]:


from typing import *

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

import pygame as pg

import os
import random

pg.init()


# In[2]:


def draw_board() -> pg.Surface:
	screen = pg.display.set_mode(WINDOW_SIZE)
	screen.fill(BACKGROUND_COLOR)
	box_size = (WINDOW_SIZE[0]/3, WINDOW_SIZE[1]/3)
	lines = [
		((box_size[0], 0), (box_size[0], WINDOW_SIZE[1])),
		((2*box_size[0], 0), (2*box_size[1], WINDOW_SIZE[1])),
		((0, box_size[1]), (WINDOW_SIZE[0], box_size[1])),
		((0, 2*box_size[1]), (WINDOW_SIZE[1], 2*box_size[1]))
	]
	for line in lines:
		pg.draw.line(screen, LINES_COLOR, line[0], line[1], LINE_THICKNESS)
	pg.display.update()
	return screen


# In[3]:


def generate_all_possible_states() -> List[np.ndarray]:
	
	def __get_possible_states(states, state, player):
		next_player = (player+1)%2
		for i in range(len(state)):
			if state[i] != -1:
				continue
			new_state = state.copy()
			new_state[i] = player
			if new_state in states:
				continue
			states.append(new_state)
			if new_state.count(-1) == 0:
				continue
			__get_possible_states(states, new_state, next_player)
		return states
	
	states = [[-1]*9] + __get_possible_states([], [-1]*9, 0)
	return [np.array(state).reshape((3, 3)) for state in states]


# In[4]:


def load_agent(path):
	global state_action_expectations, state_action_visits
	state_action_expectations = pd.read_csv(os.path.join(path, EXPECTATION_SAVE_NAME), index_col=0).to_numpy()
	state_action_visits = pd.read_csv(os.path.join(path, VISITS_SAVE_NAME), index_col=0).to_numpy()


# In[ ]:


def load_or_create_state_transition_model


# In[6]:


BOX_VALUES = ["O", "X", ""]
STATE_DTYPE = np.int32
current_player = 0
current_state = (np.zeros((3, 3)) - 1).astype(STATE_DTYPE)
agent_player = 0
WINDOW_SIZE = (600, 600)
BACKGROUND_COLOR = (255, 255, 255)
LINES_COLOR = (0, 0, 0)
TEXT_COLOR = (0, 0, 0)
LINE_THICKNESS = 5
FONT = pg.font.Font('freesansbold.ttf', 64)
DISPLAY_SURFACE = draw_board()
clock = pg.time.Clock()
EXPLORE_EXPLOIT_RATIO = 0.3
DISCOUNT_FACTOR = 0.7
LEARNING_RATE = 0.1
EPISODES = 1000
FPS = 1
EXPECTATION_SAVE_NAME = "expectations.csv"
VISITS_SAVE_NAME = "visits.csv"
SAVE_PATH = os.path.abspath("training/Data/TicTacToe/MonteCarlo-Mapping")
# state_transition_probability_model = 
states = generate_all_possible_states()
state_hashes = sorted([hash(str(state)) for state in states])
actions = [(i, j) for i in range(3) for j in range(3)]
action_hashes = sorted([hash(str(action)) for action in actions])
state_action_expectations = np.random.rand(len(actions), len(states))
state_action_visits = np.zeros((len(actions), len(states)))


def reset():
	global current_player, current_state
	
	current_player = 0
	current_state = np.zeros((3, 3)) - 1
	DISPLAY_SURFACE = draw_board()


# In[7]:


load_agent(SAVE_PATH)


# In[8]:


def draw_move(position: Tuple[int, int], player_label: str, surface: pg.Surface = DISPLAY_SURFACE):
	box_size = (WINDOW_SIZE[0]/3, WINDOW_SIZE[1]/3)
	text = FONT.render(player_label, True, TEXT_COLOR, BACKGROUND_COLOR)
	text_rect = text.get_rect()
	text_rect.center = (box_size[0]*(position[0]+0.5), box_size[1]*(position[1]+0.5))
	surface.blit(text, text_rect)
	pg.display.update()
	


# In[9]:


def get_valid_moves(state: np.ndarray) -> List[Tuple[int, int]]:
	print("[+]Getting Valid Moves...")
	actions: List[Tuple[int, int]] = []
	
	for i in range(3):
		for j in range(3):
			if state[i, j] == -1:
				actions.append((j, i))
	print(f"[+]Valid Moves: {actions}")
	return actions


# In[10]:


def get_box(pos: Tuple[int, int]) -> Tuple[int, int]:
	box_size = (WINDOW_SIZE[0]/3, WINDOW_SIZE[1]/3)
	box = [None, None]
	for i in range(2):
		for j in range(3):
			if box_size*j < pos[i] < box_size[i]*(j+1):
				box[i] = j
				break
	return tuple(box)


# In[11]:


def get_state_index(state: np.ndarray) -> int:
	return state_hashes.index(hash(str(state.astype(STATE_DTYPE))))


# In[12]:


def get_action_index(action: Tuple[int, int]) -> int:
	return action_hashes.index(hash(str(action)))


# In[13]:


def get_state_action_expectation(state: np.ndarray, action: Tuple[int, int]) -> float:
	
	return state_action_expectations[get_action_index(action), get_state_index(state)]


# In[14]:


def get_expected_probability(initial_state: np.ndarray, action: Tuple[int, int]) -> float:
	
	return 
	
	


# In[15]:


def get_state_action_expectation_DN(state: np.ndarray, action: Tuple[int, int]) -> float:
	
	
	
	


# In[16]:


def get_winner(state: np.ndarray) -> int or None:
	
	for i in range(3):
		if state[i, 0] == state[i, 1] == state[i, 2] != -1:
			return state[i, 0]
		if state[0, i] == state[1, i] == state[2, i] != -1:
			return state[i, 0]
	if state[0, 0] == state[1, 1] == state[2, 2] != -1:
		return state[0, 0]
	if state[0, 2] == state[1, 1] == state[2, 0] != -1:
		return state[0, 2]
	
	return None
	
	


# In[17]:


def is_game_over(state: np.ndarray) -> bool:
	
	if get_winner(state) is not None:
		return True
	
	for i in range(3):
		for j in range(3):
			if state[i, j] == -1:
				return False
	return True


# In[18]:


def get_reward(state: np.ndarray, player: int) -> float:
	winner = get_winner(state)
	if winner is None:
		return -1
	
	if winner == player:
		return 10
	
	return -10


# In[19]:


def update_state_action_value(state: np.ndarray, action: Tuple[int, int], value: float):        
	print(f"[+]Updating State Value Expectations for {state, action, value}")
	s = get_state_index(state)
	a = get_action_index(action)
	
	n = state_action_visits[a, s]
	state_action_expectations[a, s] = (state_action_expectations[a, s]*n + value)/(n+1)
	state_action_expectations[a, s] = state_action_expectations[a, s] + LEARNING_RATE*(value - state_action_expectations[a, s])                    
	state_action_visits[a, s] += 1
	print(f"[+]New Value: {state_action_expectations[a, s]}")
	
	


# In[20]:


def get_optimal_action(state: np.ndarray) -> Tuple[int, int]:
	print("[+]Getting Optimal Action...")
	valid_actions = get_valid_moves(state)
	expectations = [get_state_action_expectation(state, action) for action in valid_actions]
	max_expectation = max(expectations)
	optimal_action = valid_actions[expectations.index(max_expectation)]
	
	print(f"[+]Optimal Action: {optimal_action}")
	return optimal_action
	
	


# In[21]:


def get_user_action() -> Tuple[int, int]:
	print("[+]Getting User Action...")
	box_size = (WINDOW_SIZE[0]/3, WINDOW_SIZE[1]/3)
	
	while True:
		for event in pg.event.get():
			if event.type == pg.MOUSEBUTTONUP:
				pos = pg.mouse.get_pos()
				box = [None, None]
				for i in range(2):
					for j in range(3):
						if box_size[i]*j < pos[i] < box_size[i]*(j+1):
							box[i] = j
							break
				if current_state[box[1], box[0]] == -1:
					return tuple(box)
				return get_user_action()
		clock.tick(10)
		
	


# In[22]:


def get_random_move() -> Tuple[int, int]:
	print("[+]Getting Random Move...")
	actions: List[Tuple[int, int]] = get_valid_moves(current_state)
	return random.choice(actions)
	


# In[33]:


def get_opponent_move() -> Tuple[int, int]:
	return get_user_action()


# In[24]:


def explore() -> Tuple[int, int]:
	print("[+]Exploring...")
	actions: List[Tuple[int, int]] = get_valid_moves(current_state)
	return random.choice(actions)


# In[25]:


def exploit() -> Tuple[int, int]:
	print("[+]Exploiting...")
	return get_optimal_action(current_state)


# In[26]:


def get_agent_action() -> Tuple[int, int]:
	print("[+]Getting Agent Action...")
	instances = 10
	return random.choice([explore]*int(EXPLORE_EXPLOIT_RATIO*instances) + [exploit]*int((1-EXPLORE_EXPLOIT_RATIO)*instances))()


# In[27]:


def do_action(action: Tuple[int, int]):
	global current_player
	
	print(f"[+]Doing Action {action} for {current_player} on {pd.DataFrame(current_state)}...")
	
	if current_state[action[1], action[0]] != -1:
		return
	
	current_state[action[1], action[0]] = current_player
	draw_move(action, BOX_VALUES[current_player])
	current_player = (current_player + 1) % 2


# In[28]:


def do_time_step() -> Tuple[Tuple[int, int], float]:
	print("[+]Doing TimeStep...")
	print(f"[+]State: {pd.DataFrame(current_state)}")
	
	if current_player != agent_player:
		action = get_opponent_move()
		do_action(action)
	
	action = get_agent_action()
	print(f"[+]Agent Action: {action}")
	do_action(action)
	
	if not is_game_over(current_state):
		do_action(get_opponent_move())
	reward = get_reward(current_state, agent_player)
	print(f"[+]Reward: {reward}")
	
	return action, reward


# In[29]:


def do_episode():
	print("[+]Doing Episode...")
	episode_history: List[Tuple[np.ndarray, Tuple[int, int], float]] = []
	
	while not is_game_over(current_state):
		state = current_state.copy()
		action, reward = do_time_step()
		episode_history.append((state, action, reward))
		clock.tick(FPS)
	
	print("[+]Episode Done.")
	print(f"[+]Episode History: {episode_history}")
	print(f"[+]Updating Expectation Matrix...")
	for i, (state, action, reward) in enumerate(episode_history):
		update_state_action_value(state, action, sum([reward*(DISCOUNT_FACTOR**(i+1)) for i, (_, _, reward) in enumerate(episode_history[i:])]))
		
	


# In[30]:


def save_agent(path):
	
	if not os.path.isdir(path):
		os.mkdir(path)
	pd.DataFrame(state_action_expectations).to_csv(os.path.join(path, EXPECTATION_SAVE_NAME))
	pd.DataFrame(state_action_visits).to_csv(os.path.join(path, VISITS_SAVE_NAME))
	
	


# In[31]:


def start_training():
	for i in range(EPISODES):
		print(f"Episode: {i+1}")
		reset()
		do_episode()
		save_agent(SAVE_PATH)


# In[ ]:


start_training()


# In[ ]:




