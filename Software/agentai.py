import torch 
import random
import numpy as np 
from collections import deque
from model import Linear_QNet,QTrainer
import pyglet
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F 
import keyboard
import time
# import wandb
import math
from sum_tree import *

MAX_MEMORY = 100_000
BATCH_SIZE = 512
LR = 0.02
#0.0005, 0.001,
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




class Agent:

	def __init__(self):
		self.total_score=0
		self.plot_scores = []
		self.plot_mean_scores = []
		self.total_score = 0
		self.record = 0
		self.n_games = 1
		self.epsilon = 100
		self.gamma = 0

		# self.memory = deque(maxlen=MAX_MEMORY) 
		self.Memory = Memory(MAX_MEMORY) 

		self.model= Linear_QNet(2,256,256,4)
		self.target = Linear_QNet(2,256,256,4)
		self.LR = LR
		self.trainer= QTrainer(self.model,self.target,lr=LR,gamma=self.gamma)
		self.counter=1
		self.training_counter=False

	def get_state(self,Rdis,Ldis):
		

		state = [Rdis,Ldis]
		return np.array(state,dtype=int)


	def remember(self,state,action,reward,next_state, done):

		self.Memory.store((state,action,reward,next_state, done))


	def train_long_memory(self):
		tree_idx,minibatch = self.Memory.sample(BATCH_SIZE)


		states,action,rewards , next_states,dones = zip(*minibatch)
		
		self.trainer.train_step(states,action,rewards,next_states, dones,self.Memory,tree_idx)
		print("trained")



	def get_action(self,state,rewardlevel):

		explore_probability = 0.01 + (1 - 0.01) * np.exp(-0.0005 * self.n_games)

		final_move = 4

		if explore_probability > np.random.rand():

		    final_move =  random.randrange(4)


		else:
			final_move = torch.argmax(self.model(torch.tensor(state,dtype=torch.float)))



		return final_move


agent = Agent()
def train(Rdis,Ldis,action,rewardlevel):


	# if agent.counter == 1:
	# 	agent.model.load_state_dict(torch.load("./model/model.pth"))
	# 	agent.model.eval()
	# 	agent.target.load_state_dict(torch.load("./model/target.pth"))
	# 	agent.target.eval()
	# 	print("loded")

	# agent.counter =2


	state_old = agent.get_state(Rdis,Ldis)
	
	if keyboard.is_pressed('o'):
			print("old ",state_old)
	final_move = agent.get_action(state_old,rewardlevel)

	return final_move,state_old
def train2(reward,death,Rdis,Ldis,action,state_old,score):
	# wandb.log({"memory length":len(agent.Memory)})
	
	state_new = agent.get_state(Rdis,Ldis)

	if death == True:
		state_new=state_old

	if keyboard.is_pressed('o'):
		print("new ",state_new)
	if keyboard.is_pressed('/'):

		print("epsilon",agent.epsilon)

	agent.remember(state_old,action,reward,state_new,death)

	if keyboard.is_pressed('e'):
		agent.model.save('model.pth')
		agent.target.save('target.pth')
		print("saved ")

	if agent.training_counter == True or keyboard.is_pressed('q') == True:
		agent.train_long_memory()
		agent.training_counter=False

	
	

	if death == True:

		if agent.n_games%20 == 0:
			agent.epsilon-=5
			agent.training_counter=True
		agent.n_games+=1
		if score > agent.record:
			agent.model.save('model.pth')
			agent.target.save('target.pth')
			agent.record=score
		


		print('Game',agent.n_games,'Score',score,'record',agent.record,'reward',reward)

		# wandb.log({"dead reward":reward})
		# wandb.log({"score":score})

		agent.total_score+=score
		

		reward=0
		score=0



	


	




	
	


	
if __name__ == '__main__':
	train()
	
