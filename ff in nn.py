import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, input_data):
        # Input to hidden layer
        hidden_input = np.dot(input_data, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self.sigmoid(hidden_input)

        # Hidden to output layer
        output_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        final_output = self.sigmoid(output_input)

        return final_output

# Example usage:
input_size = 3
hidden_size = 4
output_size = 2

# Create a neural network with specified input, hidden, and output sizes
model = NeuralNetwork(input_size, hidden_size, output_size)

# Input data
input_data = np.array([[0.1, 0.2, 0.3]])

# Forward pass
output = model.forward(input_data)

print("Input data:", input_data)
print("Output:", output)
