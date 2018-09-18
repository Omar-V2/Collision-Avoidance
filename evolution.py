import numpy as np
import random
import matplotlib.pyplot as plt
from Sensor import Robot


class Darwin:
	def __init__(self, robot_array, elitism, mutation_rate, population_size):
		self.robot_array = robot_array
		self.generation = 0
		self.population_size = population_size
		self.elitism = elitism
		self.mutation_rate = mutation_rate
		self.best_fitness = 0
		self.number_of_parents = 4
		self.dead_count = 0
		self.x_data = []
		self.y_data = []
		self.conv_data = []

	def check_if_all_dead(self):
		self.dead_count = 0
		for robot in self.robot_array:
			if not robot.alive:
				self.dead_count += 1
		if self.dead_count == len(self.robot_array):
			return True

	def choose_parents(self):
		self.robot_array.sort(key=lambda x: x.fitness)
		max_fitness = max([robot.fitness for robot in self.robot_array])
		if max_fitness > self.best_fitness:
			self.best_fitness = max_fitness
		print("Highest fitness is: {}".format(self.best_fitness))
		return self.robot_array[(self.population_size - self.elitism):]

	def convert_to_genome(self, weights_array):
		return np.concatenate([np.ravel(i) for i in weights_array])

	def convert_to_weight(self, genome, weights_array):
		shapes = [np.shape(i) for i in weights_array]
		products = ([(i[0] * i[1]) for i in shapes])
		out = []
		start = 0
		for i in range(len(products)):
			out.append(np.reshape(genome[start:sum(products[:i + 1])], shapes[i]))
			start += products[i]
		return out

	def create_child(self, parent1, parent2):  # Uniform crossover
		parent1_genome = self.convert_to_genome(
			parent1.brain.weights)  # but is DNA actually being updated over time (over the generations)
		parent2_genome = self.convert_to_genome(parent2.brain.weights)
		child_genome = []
		for i in range(len(parent1_genome)):
			if random.random() > 0.5:
				child_genome.append(parent1_genome[i])
			else:
				child_genome.append(parent2_genome[i])
		child_weights = self.convert_to_weight(child_genome, parent1.brain.weights)
		return Robot(50, 300, 8, 360, 9, all, child_weights, own_weights=True)

	def mutate(self, individual):
		genome = self.convert_to_genome(individual.brain.weights)
		weight_to_mutate = random.randint(0, len(genome) - 1)
		genome[weight_to_mutate] = np.random.randn()
		new_weights = self.convert_to_weight(genome, individual.brain.weights)
		return Robot(50, 300, 8, 360, 9, all, new_weights, own_weights=True)

	def make_next_generation(self):
		breeders = self.choose_parents()
		offspring = []  # need to include breeders in offspring list i.e parents from nth gen should be in the n+1th generation
		for i in range(int(len(breeders) / 2)):
			for j in range(int(25)):
				offspring.append(self.create_child(breeders[i], breeders[
					len(breeders) - 1 - i]))  # make best parents breed with each other
		mutation_count = 0
		for i in range(len(offspring)):
			if random.uniform(0, 1) <= self.mutation_rate:
				offspring[i] = self.mutate(offspring[i])
				mutation_count += 1
		print("Breeders:", len(breeders))
		print("Offspring:", len(offspring))
		self.robot_array = offspring
		self.generation += 1
		print("Generation ", self.generation)

	def get_stats(self):
		all_fitness = []
		num_conv = 0
		for robot in self.robot_array:
			all_fitness.append(robot.fitness)
			if robot.hitTarget:
				num_conv += 1
		avg_fitness = sum(all_fitness) / self.population_size
		self.x_data.append(self.generation)
		self.y_data.append(avg_fitness)
		self.conv_data.append(num_conv)

	def plot_graph(self):
		plt.plot(self.x_data, self.y_data)
		plt.title("Average Fitness vs Generation")
		plt.xlabel("Generation")
		plt.ylabel("Average Fitness")
		print(self.x_data, self.y_data)
		print(self.conv_data)
		plt.show()

	def plot_graph2(self):
		plt.plot(self.x_data, self.conv_data)
		plt.title("Number of Convergences vs Generation")
		plt.xlabel("Generation")
		plt.ylabel("Number of Convergences")
		print(self.x_data, self.y_data)
		print(self.conv_data)
		plt.show()
