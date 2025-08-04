import numpy as np
import random
import matplotlib.pyplot as plt

# ===================== Data Loaders =====================
def load_flood_data(filename='Flood_Data.txt'):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # ข้าม 2 บรรทัดแรก (header)
    data_lines = lines[2:]

    dataset = []
    for line in data_lines:
        line = line.strip()
        if not line:
            continue  # ข้ามบรรทัดว่าง
        parts = list(map(float, line.split()))
        if len(parts) == 9:  # 8 inputs + 1 target
            input_data = np.array(parts[:8])
            target_data = np.array([parts[8]])
            dataset.append((input_data, target_data))

    return dataset

def load_cross_data(filename='cross.txt'):
    dataset = []
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    i = 0
    while i < len(lines):
        if lines[i].startswith('p'):
            inputs = np.array(list(map(float, lines[i+1].split())))
            outputs = np.array(list(map(int, lines[i+2].split())))
            dataset.append((inputs, outputs))
            i += 3
        else:
            i += 1
    return dataset

# ===================== Normalization =====================
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


def normalize_classification_data(data):
    inputs = np.array([x[0] for x in data])
    input_mins, input_maxs = inputs.min(axis=0), inputs.max(axis=0)
    norm_inputs = (inputs - input_mins) / (input_maxs - input_mins + 1e-8)
    return [(x, y) for x, (_, y) in zip(norm_inputs, data)]

# ===================== MLP ===============================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(output):
    return output * (1 - output)

class MLP:
    def __init__(self, num_inputs, hidden_layers, num_outputs, learning_rate, momentum):
        layer_sizes = [num_inputs] + hidden_layers + [num_outputs]
        self.weights = [np.random.uniform(-1, 1, (layer_sizes[i+1], layer_sizes[i])) for i in range(len(layer_sizes)-1)]
        self.biases = [np.random.uniform(-1, 1, (layer_sizes[i+1], 1)) for i in range(len(layer_sizes)-1)]
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
        deltas = [None] * len(self.weights)
        error = expected - self.activations[-1]
        deltas[-1] = error * sigmoid_derivative(self.activations[-1])
        for l in reversed(range(len(deltas)-1)):
            deltas[l] = np.dot(self.weights[l+1].T, deltas[l+1]) * sigmoid_derivative(self.activations[l+1])
        for l in range(len(self.weights)):
            delta_w = self.learning_rate * np.dot(deltas[l], self.activations[l].T) + self.momentum * self.prev_deltas_w[l]
            delta_b = self.learning_rate * deltas[l] + self.momentum * self.prev_deltas_b[l]
            self.weights[l] += delta_w
            self.biases[l] += delta_b
            self.prev_deltas_w[l] = delta_w
            self.prev_deltas_b[l] = delta_b

    def train(self, training_data, max_epochs=1000, target_error=0.001):
        epoch_mse_list = []
        for epoch in range(max_epochs):
            random.shuffle(training_data)
            total_error = 0
            for inputs, expected in training_data:
                outputs = self.feedforward(inputs)
                self.backpropagate(expected)
                total_error += np.sum((expected - outputs) ** 2)
            avg_error = total_error / len(training_data)
            epoch_mse_list.append(avg_error)
            if avg_error < target_error:
                break
        return epoch_mse_list

    def predict(self, inputs):
        return self.feedforward(inputs)

# ===================== Evaluation =====================
def k_fold_cross_validation(data, k=10):
    random.shuffle(data)
    fold_size = len(data) // k
    for i in range(k):
        test = data[i * fold_size:(i + 1) * fold_size]
        train = data[:i * fold_size] + data[(i + 1) * fold_size:]
        yield train, test

def mean_squared_error(dataset, model):
    errors = [
        (model.predict(x)[0] - y[0]) ** 2
        for x, y in dataset
    ]
    return np.mean(errors)

def classification_accuracy(dataset, model):
    correct = sum(np.argmax(model.predict(x)) == np.argmax(y) for x, y in dataset)
    return correct / len(dataset)

def confusion_matrix(dataset, model, num_classes=2):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for x, y in dataset:
        pred = np.argmax(model.predict(x))
        actual = np.argmax(y)
        matrix[actual][pred] += 1
    return matrix

# ===================== Main =====================
if __name__ == '__main__':
    if input("Do you want to change the default value? (y/n): ").lower() == 'y':
        k = int(input("k-fold-validation = "))
        learning_rate = float(input("learning rate = "))
        momentum_rate = float(input("momentum rate = "))
        max_epoch = int(input("Max Epoch = "))
        av_error = float(input("AV error = "))
        data_type = input("data type (flood / cross) = ").strip().lower()
        # รับ hidden layers เป็น string เช่น "3 5" แปลว่า 2 hidden layers ขนาด 3 กับ 5
        hidden_layers_input = input("Hidden layers (space separated, e.g. '3 5'): ").strip()
        if hidden_layers_input:
            hidden_layers = list(map(int, hidden_layers_input.split()))
        else:
            # กำหนด default ถ้าไม่กรอก
            hidden_layers = [10, 10]
    else:
        k, learning_rate, momentum_rate = 10, 0.01, 0.9
        max_epoch, av_error, data_type = 1000, 0.001, 'flood'
        hidden_layers = [10,10]

    # ใช้ hidden_layers ในการสร้าง model
    print(f"Data type used: '{data_type}'")
    if data_type == 'flood':
        data = load_flood_data()
        normalized_data, tmin, tmax = normalize_flood_data(data)
        total_mse = 0.0
        mse_list = []
        epoch_lists_per_fold = []
        for fold, (train, test) in enumerate(k_fold_cross_validation(normalized_data, k=k), 1):
            model = MLP(8, hidden_layers, 1, learning_rate, momentum_rate)
            mse_per_epoch = model.train(train, max_epochs=max_epoch, target_error=av_error)
            model.train(train, max_epochs=max_epoch, target_error=av_error)
            epoch_lists_per_fold.append(mse_per_epoch)
            mse = mean_squared_error(test, model)
            total_mse += mse
            mse_list.append(mse)
            print(f"[Fold {fold}] Final MSE on Test Set: {mse:.6f}, Epochs: {len(mse_per_epoch)}")
        print(f"Average MSE over {k} folds: {total_mse / k:.6f}")
        # ===== Plot MSE Graph =====
        plt.figure(figsize=(10, 6))
        for fold_index, mse_list_per_epoch in enumerate(epoch_lists_per_fold, start=1):
            plt.plot(mse_list_per_epoch, label=f'Fold {fold_index}')
        plt.title('MSE per Epoch for Each Fold')
        plt.xlabel('Epoch')
        plt.ylabel('Training MSE')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    # แสดง predict กับ actual ของ test set
            # print("Predictions vs Actual:")
            # for x, y in test:
            #     pred = model.predict(x)[0] * (tmax - tmin) + tmin
            #     actual = y[0] * (tmax - tmin) + tmin
            #     print(f"Predicted: {pred:.4f}, Actual: {actual:.4f}")


    elif data_type == 'cross':
        data = load_cross_data()
        normalized_data = normalize_classification_data(data)
        total_acc = 0.0
        for fold, (train, test) in enumerate(k_fold_cross_validation(normalized_data, k=k), 1):
            model = MLP(2, hidden_layers, 2, learning_rate, momentum_rate)
            model.train(train, max_epochs=max_epoch, target_error=av_error)
            acc = classification_accuracy(test, model)
            total_acc += acc
            cm = confusion_matrix(test, model)
            print(f"[Fold {fold}] Accuracy: {acc:.4f}")
            print("Confusion Matrix:")
            print(cm)
        print(f"Average Accuracy over {k} folds: {total_acc / k:.4f}")



else:
        print("Invalid data type. Please enter 'flood' or 'cross'.")
