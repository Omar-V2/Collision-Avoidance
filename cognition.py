import random
import numpy as np


class NeuralNetwork:
	def __init__(self, inputs, hidden_layers, hidden_neurons, outputs, given_weights):
		self.inputs = inputs
		self.hidden_layers = hidden_layers
		self.hidden_neurons = hidden_neurons
		self.outputs = outputs
		self.given_weights = given_weights
		self.weights = given_weights

	def activation(self, z, logistic=False, htan=False):
		if logistic:
			return 1 / (1 + np.exp(-z))
		elif htan:
			return np.tanh(z)

	def forward(self, x):
		new_x = self.activation(np.dot(x, self.weights[0]), htan=True)
		for i in self.weights[1:]:
			new_x = np.dot(new_x, i)
			new_x = self.activation(new_x, htan=True)
		return new_x

def convert_to_genome(weights_array):
	return np.concatenate([np.ravel(i) for i in a])

def convert_to_genome_2(weights_array):
	return np.ndarray.flatten(weights_array)


def convert_to_weights(genome, weights_array):
	shapes = [np.shape(i) for i in weights_array]
	# print(shapes)
	products = ([(i[0] * i[1]) for i in shapes])
	# print (products)
	out = []
	start = 0
	for i in range(len(products)):
		# print(sum(products[:i+1]))
		# print(start, sum(products[:i+1]))
		out.append(np.reshape(genome[start:sum(products[:i + 1])], shapes[i]))
		start += products[i]
	return out


g = np.random.randn(3,3,2)
if __name__ == '__main__':
	nn = NeuralNetwork(inputs=4, hidden_layers=1, hidden_neurons=3, outputs=2, given_weights=g)
	a = nn.weights
	print(a)
	# print(nn.weights)
	# nn.weights = [np.array([[-0.18569203, -1.15955254,  0.98317866],
	#   [-0.6967923 ,  0.68024923, -0.15005902],
	#   [ 0.6443928 , -2.061897  ,  1.10600649],
	#   [ 1.59231743, -2.21725663, -1.20812529]]), np.array([[-0.21087811, -0.59785633],
	#   [-0.07869691,  0.04648555],
	#   [-1.10778754,  1.7161734 ]])]
	# print(a)
	# print(nn.forward([21, 34, 55, 12]))
	# print(nn.forward([21, 34, 55, 12]))
	print("-------------")
	# genome = convert_to_genome(a)
	genome = convert_to_genome_2(a)
	weight = convert_to_weights(genome, weights_array=a)
	print(weight)
	# print(genome[random.randint(0, len(genome))])
	# print(genome)
	# weights = convert_to_weights(genome, a)
	# print(weights)
	# print(np.array_equal(a, weights))
# print(np.array_equal(weights[0], a[0]))
# print(np.array_equal(weights[1], a[1]))
# print(np.array_equal(weights[2], a[2]))
# print(a is type(weights))
# 		print(len(a))
# 		shapes = ([np.shape(i) for i in a])
# 		# print(shapes)
# 		# functools.reduce(lambda (x: x*y, [np.shape(i) for i in a])
# 		print([(i[0]*i[1]) for i in shapes])
# 		print(a)
#
# 		print([np.ravel(i) for i in a])
# 		b = (np.concatenate([np.ravel(i) for i in a]))
# 		print(b)
# 		# print(np.shape(b))
# 		# c = np.reshape(b[0:(9*16)], (9,16))
# 		# d = np.reshape(b[(9*16):(16*16)+(9*16)], (16,16))
# 		# e = np.reshape(b[(9*16)+(16*16):], (16, 2))
# 		# print(a[0])
# 		# print(np.array_equal(c, a[0]))
# 		# print(np.array_equal(d, a[1]))
# 		# print(np.array_equal(e, a[2]))
# # print(nn.forward([55,0.03,0.3,9]))

# array([-0.84544668,  0.79015025])
