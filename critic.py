import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense

class Critic:
	def __init__(self):
		print("Critic network initiated")

	def create_network(self, input_size, output_size, hidden_layers, perceptrons_count=64):
		model = Sequential()
		model.add(InputLayer(input_size))
		for i in range(hidden_layers):
			model.add(Dense(perceptrons_count, activation='relu'))
		model.add(Dense(output_size, activation='linear'))
		return model
	



