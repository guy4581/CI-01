import random
import math
import copy

# อ่านข้อมูล Flood_Data.txt
def load_flood_data(filename='Flood_Data.txt'):
    dataset = []
    with open(filename, 'r') as f:
        lines = f.readlines()[2:]  # ข้าม 2 บรรทัดแรก
        for line in lines:
            if line.strip() == "":
                continue
            try:
                parts = list(map(float, line.strip().split()))
                if len(parts) != 9:
                    continue  # ข้ามแถวที่มีข้อมูลไม่ครบ
                inputs = parts[:-1]  # 8 ตัวแรกเป็น input
                target = [parts[-1]]  # ตัวสุดท้ายเป็น output
                dataset.append((inputs, target))
            except ValueError:
                continue  # ข้ามถ้าแปลง float ไม่ได้
    return dataset


# อ่านข้อมูล cross.txt หรือ cross.pat
def load_cross_data(filename = 'cross.txt'):
    dataset = []
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        i = 0
        while i < len(lines):
            if lines[i].startswith('p'):
                inputs = list(map(float, lines[i+1].split()))
                outputs = list(map(int, lines[i+2].split()))
                dataset.append((inputs, outputs))
                i += 3
            else:
                i += 1  # ข้ามแถวผิดรูปแบบ
    return dataset

def normalize_flood_data(data):
    inputs = [x[0] for x in data]
    targets = [x[1][0] for x in data]

    transposed = list(zip(*inputs))
    input_mins = [min(col) for col in transposed]
    input_maxs = [max(col) for col in transposed]

    target_min = min(targets)
    target_max = max(targets)

    normalized_data = []
    for input_vec, target in data:
        norm_input = [(x - mn) / (mx - mn) if mx != mn else 0.0
                      for x, mn, mx in zip(input_vec, input_mins, input_maxs)]
        norm_target = [(target[0] - target_min) / (target_max - target_min) if target_max != target_min else 0.0]
        normalized_data.append((norm_input, norm_target))

    return normalized_data, target_min, target_max

def normalize_classification_data(data):
    inputs = [x[0] for x in data]
    transposed = list(zip(*inputs))
    input_mins = [min(col) for col in transposed]
    input_maxs = [max(col) for col in transposed]

    normalized_data = []
    for input_vec, target in data:
        norm_input = [(x - mn) / (mx - mn) if mx != mn else 0.0
                      for x, mn, mx in zip(input_vec, input_mins, input_maxs)]
        normalized_data.append((norm_input, target))

    return normalized_data

# ===================== Activation Function =====================
def sigmoid(x): return 1 / (1 + math.exp(-x))
def sigmoid_derivative(output): return output * (1 - output)

# ===================== Layer Structure =====================
class Neuron:
    def __init__(self, num_inputs):
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs + 1)]  # +1 for bias
        self.output = 0.0
        self.delta = 0.0

class Layer:
    def __init__(self, num_neurons, num_inputs_per_neuron):
        self.neurons = [Neuron(num_inputs_per_neuron) for _ in range(num_neurons)]

# ===================== MLP Network =====================
class MLP:
    def __init__(self, num_inputs, hidden_layers, num_outputs, learning_rate, momentum):
        layer_structure = [num_inputs] + hidden_layers + [num_outputs]
        self.layers = [Layer(layer_structure[i + 1], layer_structure[i]) for i in range(len(layer_structure) - 1)]
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.prev_deltas = self.initialize_prev_deltas()

    def initialize_prev_deltas(self):
        return [[[0.0 for _ in neuron.weights] for neuron in layer.neurons] for layer in self.layers]

    def feedforward(self, inputs):
        for layer in self.layers:
            new_inputs = []
            for neuron in layer.neurons:
                activation = sum(w * i for w, i in zip(neuron.weights[:-1], inputs)) + neuron.weights[-1]  # bias
                neuron.output = sigmoid(activation)
                new_inputs.append(neuron.output)
            inputs = new_inputs
        return inputs

    def backpropagate(self, expected):
        # Output layer
        for i, neuron in enumerate(self.layers[-1].neurons):
            error = expected[i] - neuron.output
            neuron.delta = error * sigmoid_derivative(neuron.output)

        # Hidden layers
        for l in reversed(range(len(self.layers) - 1)):
            for i, neuron in enumerate(self.layers[l].neurons):
                error = sum(next_neuron.weights[i] * next_neuron.delta for next_neuron in self.layers[l+1].neurons)
                neuron.delta = error * sigmoid_derivative(neuron.output)

    def update_weights(self, inputs):
        for l, layer in enumerate(self.layers):
            input_to_use = inputs if l == 0 else [n.output for n in self.layers[l - 1].neurons]
            for j, neuron in enumerate(layer.neurons):
                for k in range(len(input_to_use)):
                    delta = self.learning_rate * neuron.delta * input_to_use[k] + self.momentum * self.prev_deltas[l][j][k]
                    neuron.weights[k] += delta
                    self.prev_deltas[l][j][k] = delta
                # Update bias
                delta = self.learning_rate * neuron.delta + self.momentum * self.prev_deltas[l][j][-1]
                neuron.weights[-1] += delta
                self.prev_deltas[l][j][-1] = delta

    def train(self, training_data, max_epochs=1000, target_error=0.001):
        for epoch in range(max_epochs):
            total_error = 0.0
            for inputs, expected in training_data:
                outputs = self.feedforward(inputs)
                self.backpropagate(expected)
                self.update_weights(inputs)
                total_error += sum((expected[i] - outputs[i]) ** 2 for i in range(len(expected)))
            if total_error / len(training_data) < target_error:
                break

    def predict(self, inputs):
        return self.feedforward(inputs)

# ===================== Cross Validation =====================
def k_fold_cross_validation(data, k=10):
    random.shuffle(data)
    fold_size = len(data) // k
    for i in range(k):
        test = data[i * fold_size:(i + 1) * fold_size]
        train = data[:i * fold_size] + data[(i + 1) * fold_size:]
        yield train, test
        
def mean_squared_error(dataset, model, tmin, tmax):
    total_error = 0
    for inputs, expected in dataset:
        output = model.predict(inputs)
        # Denormalize
        predicted = output[0] * (tmax - tmin) + tmin
        actual = expected[0] * (tmax - tmin) + tmin
        total_error += (actual - predicted) ** 2
    return total_error / len(dataset)


def classification_accuracy(dataset, model):
    correct = 0
    for inputs, expected in dataset:
        output = model.predict(inputs)
        predicted = output.index(max(output))
        actual = expected.index(max(expected))
        if predicted == actual:
            correct += 1
    return correct / len(dataset)

def confusion_matrix(dataset, model, num_classes=2):
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for inputs, expected in dataset:
        output = model.predict(inputs)
        predicted = output.index(max(output))
        actual = expected.index(max(expected))
        matrix[actual][predicted] += 1
    return matrix

# ===================== Main Function =====================
if __name__ == '__main__':
    checkdata = load_flood_data('Flood_Data.txt')
    print(checkdata)
    print("Do you want to change the default value? (y/n): ", end='')
    if input().lower() == 'y':
        k = int(input("k-fold-varidation = "))
        learning_rate = float(input("learning rate = "))
        momentum_rate = float(input("momentum rate = "))
        max_epoch = int(input("Max Epoch = "))
        av_error = float(input("AV error = "))
        data_type = input("data type (flood / cross) = ").strip().lower()
    else:
        k = 10
        learning_rate = 0.3
        momentum_rate = 0.8
        max_epoch = 1000
        av_error = 0.001
        data_type = 'flood'

    if data_type == 'flood':
        data = load_flood_data('Flood_Data.txt')
        normalized_data, tmin, tmax = normalize_flood_data(data)

        total_mse = 0.0
        for fold, (train, test) in enumerate(k_fold_cross_validation(normalized_data, k=k), start=1):
            model = MLP(
                num_inputs=8,
                hidden_layers=[6],
                num_outputs=1,
                learning_rate=learning_rate,
                momentum=momentum_rate
            )
            model.train(train, max_epochs=max_epoch, target_error=av_error)

            mse = mean_squared_error(test, model, tmin, tmax)
            total_mse += mse
            print(f"[Fold {fold}] MSE: {mse:.6f}")

        print(f"\nAverage MSE ({k}-fold): {total_mse / k:.6f}")

    elif data_type == 'cross':
        data = load_cross_data('cross.txt')
        normalized_data = normalize_classification_data(data)

        total_acc = 0.0
        for fold, (train, test) in enumerate(k_fold_cross_validation(normalized_data, k=k), start=1):
            model = MLP(
                num_inputs=2,
                hidden_layers=[4],
                num_outputs=2,
                learning_rate=learning_rate,
                momentum=momentum_rate
            )
            model.train(train, max_epochs=max_epoch, target_error=av_error)

            acc = classification_accuracy(test, model)
            total_acc += acc
            cm = confusion_matrix(test, model)
            print(f"[Fold {fold}] Accuracy: {acc:.4f}")
            print("Confusion Matrix:")
            for row in cm:
                print(row)

        print(f"\nAverage Accuracy ({k}-fold): {total_acc / k:.4f}")

    else:
        print("Invalid data type. Please enter 'flood' or 'cross'.")