import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import os
import torch.onnx as onnx
import torchvision.models as models
import keyboard
import copy
import wandb
import math
from sum_tree import *
import numpy as np
counter = 1
class Linear_QNet(nn.Module):
	def __init__(self, input_size, hidden_size1,hidden_size2, output_size):
		super().__init__()
		self.linear1 = nn.Linear(input_size, hidden_size1)
		self.linear2 = nn.Linear(hidden_size1, hidden_size2)
		self.linear3 = nn.Linear(hidden_size2, output_size)
	

	def forward(self,x):
		# drop = torch.nn.Dropout(p=0.5, inplace=False)
		if keyboard.is_pressed('b'):
			print("final",x)
		
		x = self.linear1(x)
		# x = drop(x)
		x = F.relu(x)
		
		x = self.linear2(x)
		# x = drop(x)
		x = F.relu(x)

		x = self.linear3(x)
		

		if keyboard.is_pressed('a'):
			print("final",x)
		
		return x


	def save(self,file_name):
		print("file created")
		model_folder_path = './model'
		if not os.path.exists(model_folder_path):
			os.makedirs(model_folder_path)

		file_name = os.path.join(model_folder_path,file_name)

		torch.save(self.state_dict(),file_name)


	def load(self,model_path,model):
		model_folder_path = model_path
		if os.path.exists(model_folder_path):
			model.load_state_dict(torch.load(model_folder_path))
		model.eval()



class QTrainer:
	def __init__(self,model,target,lr,gamma):
		# self.lr = lr
		self.gamma = gamma
		self.model = model
		self.target = target
		self.optimizer = optim.Adam(self.model.parameters(),lr=lr)
		self.criterion = nn.MSELoss()
		



	def train_step(self,state,action,reward,next_state,done,memory,tree_idx):
		global counter

		state = torch.tensor(state, dtype=torch.float)
		next_state = torch.tensor(next_state, dtype=torch.float)
		action = torch.tensor(action, dtype=torch.long)
		reward = torch.tensor(reward, dtype=torch.float)



		if len(state.shape) == 1:
			state = torch.unsqueeze(state,0)
			next_state = torch.unsqueeze(next_state,0)
			action = torch.unsqueeze(action,0)
			reward = torch.unsqueeze(reward,0)
			done = (done, )


		targetV = self.model(state)
		target_old_clone = targetV.clone()

		target_old = target_old_clone.detach().numpy()
		
		target_next = self.model(next_state)

		target_val = self.target(next_state)


		if counter == 12:

			self.target = copy.deepcopy(self.model)
			self.target.eval()
			print("target changed")
			counter = 0


		# wandb.log({"target_counter":counter})
		counter+=1
		
		for i in range(len(done)):
		    if done[i]:
		    	targetV[i][action[i]] = reward[i]
		    else:
		        a = torch.argmax(target_next[i])
		        targetV[i][action[i]] = reward[i] + self.gamma * (target_val[i][a])  



		idx = np.arange(512, dtype=np.int32)


		target_V_clone = targetV.clone()
		targetV = targetV.detach().numpy()


		absolute_errors = np.abs(target_old[idx, np.array(action)]-targetV[idx, np.array(action)])
		memory.batch_update(tree_idx,absolute_errors)

			

		self.optimizer.zero_grad()


		# loss = self.criterion(target_old_clone,target_V_clone)
		loss = self.criterion(target_V_clone,target_old_clone)


		loss.backward()

		self.optimizer.step() 
		# wandb.log({"loss":loss})

