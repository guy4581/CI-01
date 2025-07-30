import numpy as np
import random

# ===================== อ่านและเตรียมข้อมูล =====================
def read_flood_data_file(filename='Flood_Data.txt'):
    with open(filename, 'r') as f:
        lines = f.readlines()[2:]  # ข้าม 2 บรรทัด header
    dataset = []
    for line in lines:
        if not line.strip(): continue
        parts = list(map(float, line.strip().split()))
        if len(parts) == 9:
            inputs = np.array(parts[:8])
            target = np.array([parts[8]])
            dataset.append((inputs, target))
    return dataset

def normalize_flood_data(data):
    inputs = np.array([x[0] for x in data])
    targets = np.array([x[1][0] for x in data])

    input_mins = inputs.min(axis=0)
    input_maxs = inputs.max(axis=0)
    target_min = targets.min()
    target_max = targets.max()

    input_range = input_maxs - input_mins
    input_range[input_range == 0] = 1e-8  # ป้องกันหารศูนย์

    norm_inputs = (inputs - input_mins) / input_range
    norm_targets = (targets - target_min) / (target_max - target_min + 1e-8)

    norm_data = [(x, np.array([y])) for x, y in zip(norm_inputs, norm_targets)]
    return norm_data, target_min, target_max

# ===================== Activation =====================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(y):
    return y * (1 - y)

# ===================== MLP =====================
class MLP:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.3, momentum=0.8):
        layer_sizes = [input_size] + hidden_layers + [output_size]
        self.weights = [np.random.uniform(-1, 1, (layer_sizes[i+1], layer_sizes[i])) for i in range(len(layer_sizes) - 1)]
        self.biases = [np.random.uniform(-1, 1, (size, 1)) for size in layer_sizes[1:]]

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.prev_deltas_w = [np.zeros_like(w) for w in self.weights]
        self.prev_deltas_b = [np.zeros_like(b) for b in self.biases]

    def feedforward(self, inputs):
        activation = inputs.reshape(-1, 1)
        self.activations = [activation]
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            activation = sigmoid(z)
            self.activations.append(activation)
        return activation.flatten()

    def backpropagate(self, expected):
        expected = expected.reshape(-1, 1)
        error = expected - self.activations[-1]
        deltas = [error * sigmoid_derivative(self.activations[-1])]
        for l in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(self.weights[l + 1].T, deltas[0]) * sigmoid_derivative(self.activations[l + 1])
            deltas.insert(0, delta)
        for l in range(len(self.weights)):
            delta_w = self.learning_rate * np.dot(deltas[l], self.activations[l].T) + self.momentum * self.prev_deltas_w[l]
            delta_b = self.learning_rate * deltas[l] + self.momentum * self.prev_deltas_b[l]
            self.weights[l] += delta_w
            self.biases[l] += delta_b
            self.prev_deltas_w[l] = delta_w
            self.prev_deltas_b[l] = delta_b

    def train(self, data, epochs=1000, target_error=0.001):
        for epoch in range(epochs):
            random.shuffle(data)
            total_error = 0
            for x, y in data:
                output = self.feedforward(x)
                self.backpropagate(y)
                total_error += np.sum((y - output) ** 2)
            avg_error = total_error / len(data)
            if avg_error < target_error:
                print(f"Training stopped at epoch {epoch} with avg error: {avg_error:.6f}")
                break

    def predict(self, x):
        return self.feedforward(x)

# ===================== Mean Squared Error =====================
def mean_squared_error(data, model, tmin, tmax):
    errors = []
    for x, y in data:
        pred = model.predict(x)[0] * (tmax - tmin) + tmin
        actual = y[0] * (tmax - tmin) + tmin
        errors.append((pred - actual) ** 2)
    return np.mean(errors)

# ===================== ทดสอบใช้งาน =====================
if __name__ == '__main__':
    raw_data = read_flood_data_file('Flood_Data.txt')
    norm_data, tmin, tmax = normalize_flood_data(raw_data)

    # แบ่งเทรน/เทสต์ 80/20
    split = int(len(norm_data) * 0.8)
    train_data = norm_data[:split]
    test_data = norm_data[split:]

    mlp = MLP(input_size=8, hidden_layers=[10, 10], output_size=1, learning_rate=0.3, momentum=0.8)
    mlp.train(train_data, epochs=1000, target_error=0.001)

    # ประเมินผล
    mse = mean_squared_error(test_data, mlp, tmin, tmax)
    print(f"\nTest MSE: {mse:.4f}")
    print("\nPredicted vs Actual:")
    for x, y in test_data[:20]:
        pred = mlp.predict(x)[0] * (tmax - tmin) + tmin
        actual = y[0] * (tmax - tmin) + tmin
        print(f"Predicted: {pred:.2f}, Actual: {actual:.2f}")
