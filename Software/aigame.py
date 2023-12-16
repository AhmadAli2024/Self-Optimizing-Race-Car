#importing libraries
import pyglet
from pyglet.window import key
import math
from random import randint
import numpy as np
from pyglet import shapes 
from torch import nn
import torch
from collections import deque
import itertools
import random
from multiprocessing import Process
import time
from pyglet import clock
import agentai
import keyboard 
import warnings

# warnings.filterwarnings('ignore')
# wandb.init(project='Q learning car ai',entity='aaliahmad', config={
# 	"learning_rate": 0.00045,
# 	"layer_1": 8,
# 	"activation_1": "relu",
# 	"dropout": 0.2,
# 	"layer_2": 256,
# 	"activation_2": "relu",
# 	"layer_3": 256,
# 	"activation_3": "relu",
# 	"layer_2": 3,
# 	"activation_2": "softmax",
# 	"optimizer": "adam",
# 	"batch_size": 516
# })
# config = wandb.config








#creating window
game_window = pyglet.window.Window(1280, 720)
#the drawing batch
main_batch=pyglet.graphics.Batch()
#centering image code
def center_image(image):
	image.anchor_x = image.width // 2
	image.anchor_y = image.height // 2
def getmeasurments(image):
	global imagex
	global imagey
	imagex = image.width
	imagey = image.height
def lineEquation(x1,y1,x2,y2):
	gradientx = x2-x1
	gradienty = y2-y1
	gradient = gradienty/gradientx
	y_interpert = (gradient*-x1)+y1
	return gradient,y_interpert

#displaying image code
pyglet.resource.path = ['../images']
pyglet.resource.reindex()
player_image = pyglet.resource.image("Car3.png")
sensor_line = pyglet.resource.image("line.png")
center_image(player_image)
getmeasurments(sensor_line)
player_car = pyglet.sprite.Sprite(img=player_image, x=200, y=500)
Sline = pyglet.sprite.Sprite(img=sensor_line,x=200,y=20)


vertex_list = ["","","","","","","",""]


#the borders of the track

def straight_track():
	height = 144
	width = 256
	rewardwidth = width+width/2
	halfheight = height/2
	rewardheight = height+halfheight
	lines = [
			
			#original track
			[width,height,width*2,height],
			[width,height*4,width*2,height*4],

			[width,height,width+1,height*4],
			[width*2,height,width*2+1,height*4],
			[width*3,0,width*3+1,height*5],
			#lines across the screen
			[0,0,1280,0],[1,0,0,720],[0,719,1280,720],[1279,720,1280,0],

	]

	rewardlines = [[width*2,rewardheight+1,width*3,rewardheight],[rewardwidth+1,height*4,rewardwidth,1280],[0,rewardheight*2+halfheight+1,width,rewardheight*2+halfheight],[0,rewardheight+1,width,rewardheight],[rewardwidth+1,0,rewardwidth,height]]

	return lines,rewardlines
def normal_track():
	lines=[[250,250,251,500],[250,500,500,590],[500,590,1020,580],[1020,580,1050,200],[1050,200,600,400],[600,400,250,250],[300,120,150,200],
[150,200,151,600],
[150,600,500,700],[500,700,1100,700],[1100 ,700,1200,500],[1200,500,1201,100],[1200,100,900,100],[900,100,600,300],[600,300,450,200],
[450,200,300,120]]
	rewardlines=[[251,400,150,400],[251,500,150,600],[500,590,501,700],[1020,580,1200,500],[1050,200,1200,100],[600,400,601,300],[250,250,150,200]]

	return lines,rewardlines
def right_track():
	height = 144
	width = 256
	rewardwidth = width+width/2
	halfheight = height/2
	rewardheight = height+halfheight
	lines = [
			[width,height,width*2,height],
			[width*3,height*2,width*5,height*2],
			[width*2,height*3,width*4,height*3],
			[width,height*4,width*4,height*4],


			[width,height,width+1,height*4],
			[width*2,height,width*2+1,height*3],
			[width*3,0,width*3+1,height*2],
			[width*4,height*3,width*4+1,height*4],
			#lines across the screen
			[0,0,1280,0],[1,0,0,720],[0,719,1280,720],[1279,720,1280,0],

	]

	rewardlines = [width*2,rewardheight+1,width*3,rewardheight],[rewardwidth*2+1,rewardheight,rewardwidth*2,rewardheight*2],[width*4,rewardheight*2+halfheight,width*5,rewardheight*2+halfheight],[rewardwidth*2+1,height*4,rewardwidth*2,height*5],[rewardwidth+1,height*4,rewardwidth,1280],[0,rewardheight*2+halfheight+1,width,rewardheight*2+halfheight],[0,rewardheight+1,width,rewardheight],[rewardwidth+1,0,rewardwidth,height]

	return lines,rewardlines

def left_track():
	height = 144
	width = 256
	lines = [

			[0,height*2,width*2,height*2],
			[width*2+1,1,width*2,height*2],
			[width,height*3,width+1,height*4],
			[width,height*4,width*4,height*4],
			[width,height*3,width*3,height*3],
			[width*3+1,height,width*3,height*3],
			[width*4+1,height,width*4,height*4],
			[width*3,height,width*4,height],

			#lines across the screen
			[0,0,1280,0],[1,0,0,720],[0,719,1280,720],[1279,720,1280,0],

	]
	rewardwidth = width+width/2
	halfheight = height/2
	halfwidth = width/2
	rewardheight = height+halfheight
	rewardlines = [
					[width*2,rewardheight+1,width*3,rewardheight],
					[0,rewardheight*2+halfheight,width,rewardheight*2+halfheight],
					[rewardwidth+1,height*4,rewardwidth,height*5],
					[rewardwidth*2+1+halfwidth,height*4,rewardwidth*2+halfwidth,height*5],
					[width*4,rewardheight*2+halfheight,width*5,rewardheight*2+halfheight],
					[rewardwidth*2+1+halfwidth,0,rewardwidth*2+halfwidth,height],
					]

	return lines,rewardlines



lines = straight_track()[0]
rewardlines = straight_track()[1]
global tempC
global tempD
global smallestD
tempD = 0
tempC = 0
agentDeath = False
#the laws of phycics
class PhysicalObject(pyglet.sprite.Sprite):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.rotation = 0
		self.velocity_x = 0.0
		self.velocity_y = 0.0
		self.thrust = 3311
		
	def update(self, dt):

		self.x += self.velocity_x * dt
		self.y += self.velocity_y * dt





#the players code
class Player(PhysicalObject):
	
	#some variables
	if tempC == 0:
		action = 6
		tempC+=1
		reward = 0
		rewardlevel = 0
		score = 0
		dead = False
	def __init__(self, *args, **kwargs):
		super().__init__(img=player_image,*args, **kwargs)

		
		#some counters
		self.tempcounter=0
		self.deltatheta=-3
		self.counter=0


	def update(self, dt):
		super(Player, self).update(dt)
		global agentDeath
		#getting the action from agent
		


		pyglet.clock.unschedule(update)
		player_car.action = agentai.train(uleftline.smallestD,urightline.smallestD,Player.action,Player.rewardlevel)
		pyglet.clock.schedule_interval(update, 1/30)
		Player.action = player_car.action[0]



		



		angle_radians = math.radians(self.rotation)
		dx = math.sin(angle_radians+1.570796)*309
		dy = math.cos(angle_radians+1.570796)*309
		def calculating_point(g1,y_in1,g2,y_in2,minX,minY,maxX,maxY,carx,cary,Circle):
			A = np.array([[g1,1], [g2,1]])
			B = np.array([y_in1,y_in2])
			D = np.linalg.inv(A)
			E = np.dot(D,B)
			E[0] = E[0]*-1
			Exmax = E[0]-self.x
			Eymax = E[1]-self.y
			EMAX = Exmax+Eymax
			if minY >= maxY:
				minY,maxY=maxY,minY
			if minX >= maxX:
				minX,maxX=maxX,minX
						
			if abs(EMAX) <= Circle:
				if E[0] >= minX and E[0]<=maxX and E[1]>=minY and E[1] <=maxY:
					if self.rotation >=270 or self.rotation <=90:
						if E[0]<=self.x+dx and E[0]>=self.x:
							return E[0],E[1]
					if self.rotation>=90 or self.rotation<=270:
						if E[0] >=self.x+dx and E[0]<=self.x:
							return E[0],E[1]
				return 0

		def distance(x1,y1,x2,y2):
			x_sum = (x2-x1)**2
			y_sum = (y2-y1)**2
			return math.sqrt(x_sum+y_sum)
			
		def get_middle(x1,y1,x2,y2):
			return (x1+x2)/2,(y1+y2)/2


		angle_radians = math.radians(self.rotation)
		force_x = math.sin(angle_radians) * self.thrust * dt
		force_y = math.cos(angle_radians) * self.thrust * dt
		self.velocity_x=0
		self.velocity_y=0

		



		deathgradient1,deathyintercept1 = lineEquation(self.x+5,self.y+5,self.x-5,self.y-5)

		deathintersect1list=[]
		global rewardlines
		x1,y1,x2,y2=rewardlines[Player.rewardlevel%len(rewardlines)][0],rewardlines[Player.rewardlevel%len(rewardlines)][1],rewardlines[Player.rewardlevel%len(rewardlines)][2],rewardlines[Player.rewardlevel%len(rewardlines)][3]
		x11,y11,x22,y22 = rewardlines[Player.rewardlevel%len(rewardlines)-1][0],rewardlines[Player.rewardlevel%len(rewardlines)-1][1],rewardlines[Player.rewardlevel%len(rewardlines)-1][2],rewardlines[Player.rewardlevel%len(rewardlines)-1][3]
		# print(x1,y1,x2,y2)
		gradient,yintercept = lineEquation(x1,y1,x2,y2)
	
		rewardintercept = calculating_point(deathgradient1,deathyintercept1,gradient,yintercept,x1,y1,x2,y2,self.x,self.y,30)


		# middlex,middley = get_middle(x1,y1,x2,y2)
		# middlex2,middley2 = get_middle(x11,y11,x22,y22)
		# Player.reward+= (distance(middlex2,middley2,middlex,middley) - distance(self.x,self.y,middlex,middley))/50
		# print(distance(middlex2,middley2,middlex,middley) - distance(self.x,self.y,middlex,middley)/3)


		global lines
		if rewardintercept:
			if Player.rewardlevel%len(rewardlines) == 0:
				Player.score-=1
				Player.reward-=1000
				random = randint(0,2)
				# wandb.log({"track":random})
				if random == 0:
					lines = straight_track()[0]
					rewardlines = straight_track()[1]
				elif random == 1:
					lines = right_track()[0]
					rewardlines = right_track()[1]
				else:
					lines = left_track()[0]
					rewardlines = left_track()[1]
			print("epic")
			Player.rewardlevel+=1
			Player.score+=1
			Player.reward+=1000


		global vertex_list
		for coordinates in lines:
			x1,y1,x2,y2=coordinates[0],coordinates[1],coordinates[2],coordinates[3]
			
			gradient,yintercept = lineEquation(x1,y1,x2,y2)

			
			deathintersect1 = calculating_point(deathgradient1,deathyintercept1,gradient,yintercept,x1,y1,x2,y2,self.x,self.y,15)


			if deathintersect1:
				Player.dead=True

		if self.x > 1280 or self.x < 0 or self.y >720 or self.y < 0:
			Player.dead = True

		if Player.dead:
			print("dead")
			self.x=650
			self.y=100
			self.rotation=0
			Player.rewardlevel=0
			Player.reward-=2000
			agentDeath = True


		Player.dead = False
		

		

		

		#actions the agent can take
		if player_car.action[0] == 0:
			self.velocity_x = force_x 
			self.velocity_y = force_y 

		if player_car.action[0]==1:
			self.velocity_x = force_x 
			self.velocity_y = force_y 
			self.rotation+=3.38
			
		if player_car.action[0]==2:
			self.velocity_x = force_x 
			self.velocity_y = force_y 
			self.rotation-=3.38


		

		# if keyboard.is_pressed("w"):
		# 	self.velocity_x = force_x 
		# 	self.velocity_y = force_y 

		# if keyboard.is_pressed("d"):
		# 	self.velocity_x = force_x 
		# 	self.velocity_y = force_y 
		# 	self.rotation+=4
			
		# if keyboard.is_pressed("a"):
		# 	self.velocity_x = force_x 
		# 	self.velocity_y = force_y 
		# 	self.rotation-=4
		


class Sensor(PhysicalObject):
	

	#defining variables
	if tempD == 0:
		dead= False
		score=0
		tempD+=1
		closestInter=0


	def __init__(self, *args, **kwargs):
		super().__init__(img=sensor_line,*args, **kwargs)
		
		self.smallestD = 0
		
	def update(self, dt,carx,cary,rotation,name,carr):
		super(Sensor, self).update(dt)
		
		

		angle_radians = math.radians(self.rotation)
		shiftx = math.sin(angle_radians) * 15
		shifty = math.cos(angle_radians) * 15

		self.rotation=rotation+carr
		self.x = carx 
		self.y = cary 
		dx = math.sin(angle_radians+1.570796)*309
		dy = math.cos(angle_radians+1.570796)*309
		
		sensorGraident,sensorYintercept = lineEquation(self.x,self.y,self.x+dx,self.y+dy)
		
		def calculating_point(g1,y_in1,g2,y_in2,minX,minY,maxX,maxY,carx,cary,Circle):

			A = np.array([[g1,1], [g2,1]])
			B = np.array([y_in1,y_in2])
			D = np.linalg.inv(A)
			E = np.dot(D,B)
			E[0] = E[0]*-1
			Exmax = E[0]-self.x
			Eymax = E[1]-self.y
			EMAX = Exmax+Eymax
			if minY >= maxY:
				minY,maxY=maxY,minY
			if minX >= maxX:
				minX,maxX=maxX,minX
						
			if abs(EMAX) <= Circle:
				if E[0] >= minX and E[0]<=maxX and E[1]>=minY and E[1] <=maxY:
					if self.rotation >=270 or self.rotation <=90:
						if E[0]<=self.x+dx and E[0]>=self.x:
							return E[0],E[1]
					if self.rotation>=90 or self.rotation<=270:
						if E[0] >=self.x+dx and E[0]<=self.x:
							return E[0],E[1]
				return 0
			return 0

		intersectlist=[]
		distancelist = []
			


		global vertex_list
		for coordinates in lines:

			x1,y1,x2,y2=coordinates[0],coordinates[1],coordinates[2],coordinates[3]

			gradient,yintercept = lineEquation(x1,y1,x2,y2)

			intersect= calculating_point(sensorGraident,sensorYintercept,gradient,yintercept,x1,y1,x2,y2,carx,cary,309)

			if intersect:
				intersectlist.append(intersect)
				distancelist.append(math.sqrt((self.x-intersect[0])**2+(self.y-intersect[1])**2))
				

			
		if len(intersectlist)>=1:
			closest = distancelist.index(min(distancelist))
			closestInter = intersectlist[closest]
			vertex_list[int(name)] = shapes.Circle(closestInter[0], closestInter[1],5,color=( 255,255,255),batch=main_batch)
			self.smallestD = distancelist[closest]
		if len(intersectlist) == 0:
			self.smallestD=309
		

		
	

		





		
		
		

		

						







#integrating the player
player_car = Player(x=650, y=100)
# ForwardLine = Sensor(x=300,y=50)
uleftline = Sensor(x=199,y=315)
urightline = Sensor(x=199,y=315)


game_window.push_handlers(player_car)
# game_window.push_handlers(ForwardLine)
game_window.push_handlers(uleftline)
game_window.push_handlers(urightline)

#the update function
Ttime = 0
trackcounter =0
def update(dt):
	global Ttime
	global agentDeath
	global trackcounter

	pyglet.clock.unschedule(update)
	player_car.update(dt)
	# ForwardLine.update(dt,player_car.x,player_car.y,270,"0",player_car.rotation)
	uleftline.update(dt,player_car.x,player_car.y,-65,"4",player_car.rotation)
	urightline.update(dt,player_car.x,player_car.y,245,"5",player_car.rotation)
	

	# Player.reward+= Ttime/60*1.001

	# if player_car.action[0] == 3:

	# 	Player.reward-= Ttime/60*1.001

	
	agentai.train2(Player.reward,agentDeath,uleftline.smallestD,urightline.smallestD,Player.action,player_car.action[1],Player.score)

	Ttime+=1
	if keyboard.is_pressed("]"):
		print(SensorLine.smallestD,player_car.counter)
	if keyboard.is_pressed('d'):
		print(agentDeath)
	if agentDeath:
		trackcounter+=1
		Player.score = 0
		Player.reward = 0
		Ttime=0
	
	agentDeath = False
	pyglet.clock.schedule_interval(update, 1/30)







@game_window.event
def on_draw():
	global trackcounter
	global lines 
	global rewardlines
	global trackone

	game_window.clear()

	main_batch.draw()

	for coordinates in lines:
		pyglet.graphics.draw(2, pyglet.gl.GL_LINES,
		('v2f', (coordinates[0],coordinates[1],coordinates[2],coordinates[3])))
	for coordinates in rewardlines:
		pyglet.graphics.draw(2, pyglet.gl.GL_LINES,
		('v2f', (coordinates[0],coordinates[1],coordinates[2],coordinates[3])))


	# ForwardLine.draw()
	uleftline.draw()
	urightline.draw()
	player_car.draw()




if __name__ == '__main__':
	pyglet.clock.schedule_interval(update, 1/30)
	pyglet.app.run()
	# wandb.finish()
