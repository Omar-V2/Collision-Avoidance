import pygame
from evolution import Darwin
from Sensor import Robot, obstacleArray


# Game Settings
pygame.init()
background_colour = (0, 0, 0)
(width, height) = (1000, 600)
target_location = (800, 300)
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Omar's Simulation")
screen.fill(background_colour)


# GA Hyper parameters
population_size = 50
elitism = 4

# Agent Initialisation
robots = []
for i in range(population_size):
	robots.append(Robot(175, 300, 10, 360, 9, all, set_weights=None))
darwin = Darwin(robot_array=robots, population_size=population_size, elitism=4, mutation_rate=0.1)



if __name__ == '__main__':
	running = True
	while running:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
		screen.fill(background_colour)
		pygame.draw.rect(screen, (255, 255, 255), (10, 10, width - 20, height - 20), 1)
		pygame.draw.circle(screen, (255, 10, 0), target_location, 10, 0)
		# pygame.draw.line(screen, (255, 0, 0), (800, 10), (800, 590))
		for obstacle in obstacleArray:
			obstacle.drawShape()
		# obstacle.move_y()
		# pygame.draw.circle(screen, (0, 0, 255), (500, 300), 100, 0)
		# pygame.draw.circle(screen, (0, 255, 20), (200, 300), 75, 0)
		# pygame.draw.polygon(screen, (255, 255, 255), new_list, 1)
		# for pedestrian in all.start_pedestrians:
		# 		pedestrian.move()
		# 		pedestrian.update()
		# 		all.introduce()
		for robot in darwin.robot_array:
			robot.move()
			robot.update()
			robot.collide()
			robot.evaluate_fitness()
		if darwin.check_if_all_dead():
			darwin.get_stats()
			darwin.make_next_generation()
		pygame.display.update()
