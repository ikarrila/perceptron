import numpy as np

#how many training iterations, bigger number -> more precise results, but takes time
loops = 1000

#logistic function to put inputs in order
def logistic(x):
    return 1 / (1 + np.exp(-x))

#derivative of the logisctic function is used to determine how much weights ought to be adjusted
def derivative(x):
    return x * (1 - x)

#picking random weights, creating weight matrix
weights = 2 * np.random.random((3,1)) - 1
print('Starting weights: ')
print(weights)

#inputs and correct outputs
training_outputs = np.array([[1,0,1,1,0]]).T
inputs = np.array([[1,1,1],
                    [0,0,1],
                    [1,0,1],
                    [1,0,0],
                    [0,1,1]])

#iteration loop where weights are adjusted depending on the error margin
for i in range(loops):
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
