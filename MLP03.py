import numpy as np
import random
import math

# === CONFIGURABLE PARAMETERS ===
params = {
    "k_fold": 10,
    "learning_rate": 0.3,
    "momentum": 0.8,
    "max_epoch": 1000,
    "avg_error": 0.001,
    "hidden_layers": [8],  # ex: [8] for 1 hidden layer with 8 nodes, or [8, 5] for 2 layers
    "data_type": "regression",  # or "classification"
    "seed": 42,
}

np.random.seed(params["seed"])

# === UTILITY FUNCTIONS ===
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(output):
    return output * (1.0 - output)

def normalize(data):
    data = np.array(data)
    min_v, max_v = np.min(data, axis=0), np.max(data, axis=0)
    return (data - min_v) / (max_v - min_v + 1e-8), min_v, max_v

def denormalize(data, min_v, max_v):
    return data * (max_v - min_v + 1e-8) + min_v

def one_hot_encode(y):
    classes = list(sorted(set(y)))
    encoded = []
    for label in y:
        vec = [0] * len(classes)
        vec[classes.index(label)] = 1
        encoded.append(vec)
    return np.array(encoded), classes

# === DATA LOADING ===
def load_flood_data(path='Flood_Data.txt'):
    with open(path, 'r') as f:
        lines = f.readlines()[2:]
        data = []
        for line in lines:
            if line.strip() == "": continue
            values = list(map(float, line.strip().split()))
            data.append(values)
    return np.array(data)

def load_cross_pat(path='cross.pat'):
    with open(path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
        data = []
        for line in lines:
            values = list(map(float, line.split()))
            data.append(values)
    return np.array(data)

# === NEURAL NETWORK CLASS ===
class MLP:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.deltas = []
        self.last_deltas = []
        for i in range(len(layers) - 1):
            w = np.random.uniform(-1.0, 1.0, (layers[i]+1, layers[i+1]))  # +1 for bias
            self.weights.append(w)
            self.last_deltas.append(np.zeros_like(w))

    def forward(self, x):
        inputs = x
        activations = [np.append(x, 1.0)]  # bias
        for w in self.weights:
            net_input = np.dot(activations[-1], w)
            output = sigmoid(net_input)
            activations.append(np.append(output, 1.0))  # bias
        activations[-1] = activations[-1][:-1]  # remove bias at output
        return activations

    def backward(self, activations, target, lr, momentum):
        errors = target - activations[-1]
        deltas = [errors * sigmoid_derivative(activations[-1])]
        for i in reversed(range(len(self.weights)-1)):
            w = self.weights[i+1][:-1, :]  # remove bias weights
            delta = deltas[-1].dot(w.T) * sigmoid_derivative(activations[i+1][:-1])
            deltas.append(delta)
        deltas.reverse()

        for i in range(len(self.weights)):
            a = np.atleast_2d(activations[i])
            d = np.atleast_2d(deltas[i])
            change = lr * a.T.dot(d) + momentum * self.last_deltas[i]
            self.weights[i] += change
            self.last_deltas[i] = change

        return np.mean(errors ** 2)

    def predict(self, x):
        a = self.forward(x)[-1]
        return a

# === CROSS VALIDATION ===
def k_fold_split(data, k):
    np.random.shuffle(data)
    fold_size = len(data) // k
    folds = [data[i * fold_size:(i + 1) * fold_size] for i in range(k)]
    return folds

def train_and_evaluate(data):
    folds = k_fold_split(data, params["k_fold"])
    total_mse = []
    all_preds = []
    all_true = []

    for i in range(params["k_fold"]):
        test = folds[i]
        train = np.vstack([folds[j] for j in range(params["k_fold"]) if j != i])

        X_train, y_train = train[:, :-1], train[:, -1:]
        X_test, y_test = test[:, :-1], test[:, -1:]

        if params["data_type"] == "classification":
            y_train, classes = one_hot_encode(y_train.flatten())
            y_test, _ = one_hot_encode(y_test.flatten())

        X_train, min_X, max_X = normalize(X_train)
        X_test = (X_test - min_X) / (max_X - min_X + 1e-8)

        if params["data_type"] == "regression":
            y_train, min_y, max_y = normalize(y_train)
            y_test_norm = (y_test - min_y) / (max_y - min_y + 1e-8)

        input_size = X_train.shape[1]
        output_size = y_train.shape[1] if params["data_type"] == "classification" else 1
        net = MLP([input_size] + params["hidden_layers"] + [output_size])

        for epoch in range(params["max_epoch"]):
            total_error = 0
            for xi, yi in zip(X_train, y_train):
                activations = net.forward(xi)
                total_error += net.backward(activations, yi, params["learning_rate"], params["momentum"])
            if total_error / len(X_train) < params["avg_error"]:
                break

        # prediction
        preds = []
        for xt in X_test:
            pred = net.predict(xt)
            preds.append(pred)

        preds = np.array(preds)
        if params["data_type"] == "regression":
            preds = denormalize(preds, min_y, max_y)
            mse = np.mean((preds - y_test) ** 2)
            total_mse.append(mse)
        else:
            pred_class = np.argmax(preds, axis=1)
            true_class = np.argmax(y_test, axis=1)
            all_preds += list(pred_class)
            all_true += list(true_class)

    if params["data_type"] == "regression":
        print(f"Average MSE = {np.mean(total_mse):.6f}")
    else:
        report_confusion(all_true, all_preds)

def report_confusion(y_true, y_pred):
    from collections import Counter
    size = max(max(y_true), max(y_pred)) + 1
    conf = np.zeros((size, size), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf[t][p] += 1
    print("Confusion Matrix:")
    print(conf)
    acc = np.sum(np.diag(conf)) / np.sum(conf)
    print(f"Accuracy: {acc*100:.2f}%")

# === MAIN PROGRAM ===
if __name__ == "__main__":
    print("Do you want to change default parameters? (y/n): ", end="")
    if input().lower() == 'y':
        params["k_fold"] = int(input("k-fold-validation = "))
        params["learning_rate"] = float(input("learning rate = "))
        params["momentum"] = float(input("momentum rate = "))
        params["max_epoch"] = int(input("Max Epoch = "))
        params["avg_error"] = float(input("AV error = "))
        params["data_type"] = input("data type (regression/classification) = ").strip().lower()
        hidden_str = input("hidden layers (e.g. 8 or 8,5): ")
        params["hidden_layers"] = list(map(int, hidden_str.strip().split(',')))

    if params["data_type"] == "regression":
        dataset = load_flood_data()
    else:
        dataset = load_cross_pat()

    train_and_evaluate(dataset)
