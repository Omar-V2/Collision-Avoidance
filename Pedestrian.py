import pygame
import random
from ProcessData import My_Trajectory_Dict, Pedestrian_IDs


class Pedestrian:
	def __init__(self, x, y, size, id, trajectory):
		self.x = x
		self.y = y
		self.size = size
		self.id = id
		self.trajectory = trajectory
		self.colour = (random.randint(20, 255), random.randint(20, 255), random.randint(20, 255))
		self.game_width = width
		self.game_height = height
		self.present = True
		self.considered = False
		self.destination = 0
		self.paths = []

	def update(self):
		if self.present:
			pygame.draw.circle(screen, self.colour, ((int(self.x) + int(width / 2)), (int(self.y) + int(height / 2))),
			                   self.size, 0)

	def move(self):
		if self.destination < len(self.trajectory[0]):
			self.x = self.trajectory[0][self.destination]
			self.y = -self.trajectory[1][self.destination]
			self.destination += 1
			self.paths.append([self.x + (width / 2), self.y + (height / 2)])
		else:
			self.present = False


class Manager(Pedestrian):
	def __init__(self, all_pedestrians, limit):
		self.limit = limit
		self.all_pedestrians = all_pedestrians
		self.start_pedestrians = all_pedestrians[:self.limit]

	def introduce(self):
		for pedestrian in self.all_pedestrians:
			if not pedestrian.present:
				self.all_pedestrians.remove(pedestrian)
				self.start_pedestrians.remove(pedestrian)
				for i in self.all_pedestrians:
					if i not in self.start_pedestrians and len(self.start_pedestrians) <= self.limit:
						self.start_pedestrians.append(i)


all_pedestrians = []
for pedestrian in Pedestrian_IDs:
	starting_x = My_Trajectory_Dict[pedestrian][0][0]
	starting_y = My_Trajectory_Dict[pedestrian][0][1]
	all_pedestrians.append(Pedestrian(starting_x, starting_y, 8, pedestrian, My_Trajectory_Dict[pedestrian]))

all = Manager(all_pedestrians, 15)

