import numpy as np

loops = 1000

def logistic(x):
    return 1 / (1 + np.exp(-x))

def derivative(x):
    return x * (1 - x)

inputs = np.array([[1,1,1],
                    [0,0,1],
                    [1,0,1],
                    [1,0,0],
                    [0,1,1]])

training_outputs = np.array([[1,0,1,1,0]]).T

weights = 2 * np.random.random((3,1)) - 1

print('Starting weights: ')
print(weights)

for iteration in range(loops):
    input_layer = inputs
    outputs = logistic(np.dot(input_layer, weights))
    error = training_outputs - outputs
    adjust = error * derivative(outputs)
    weights += np.dot(input_layer.T, adjust)

print('Trained weights: ')
print(weights)

print("Output:")
print(outputs)
print("Correct line: {}".format(training_outputs.T))