import numpy as np
import matplotlib.pyplot as plt
import copy

class NN:
    def __init__(self, layer, learning_rate = 0.1, momentum_rate=0.9, activation_function='sigmoid'):

        self.V = []
        self.layer = layer
        self.momentum_rate = momentum_rate
        self.learning_rate = learning_rate
        self.activation_function = activation_function

        self.w, self.delta_w, self.b, self.delta_bias, self.local_gradient = self.init_inform(layer)

    def activation(self, x):
        if self.activation_function == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation_function == "relu":
            return np.where(x > 0, x, 0.0)
        elif self.activation_function == "tanh":
            return np.tanh(x)
        elif self.activation_function == "linear":
            return x
    def activation_diff(self, x):
        if self.activation_function == "sigmoid":
            return x * (1 - x)
        elif self.activation_function == "relu":
            return np.where(x > 0, 1.0, 0.0)
        elif self.activation_function == "tanh":
            return 1 - x**2
        elif self.activation_function == "linear":
            return np.ones_like(x)

    def init_inform(self, layer):
        weights = []
        delta_weights = []
        biases = []
        delta_biases = []
        local_gradientes = [np.zeros(layer[0])]
        for i in range(1, len(layer), 1):
            weights.append(np.random.rand(layer[i], layer[i-1]))
            delta_weights.append(np.zeros((layer[i], layer[i-1])))
            biases.append(np.random.rand(layer[i]))
            delta_biases.append(np.zeros(layer[i]))
            local_gradientes.append(np.zeros(layer[i]))
        return weights, delta_weights, biases, delta_biases, local_gradientes
    
    def feed_forward(self, input):
        self.V = [input]
        for i in range(len(self.layer) - 1):
            self.V.append(self.activation((self.w[i] @ self.V[i]) + self.b[i]))

    def back_propagation(self, design_output):
        for i, j in enumerate(reversed(range(1, len(self.layer), 1))):
            if i == 0:
                error = np.array(design_output - self.V[j])
                self.local_gradient[j] = error * self.activation_diff(self.V[j])
            else:
                self.local_gradient[j] = self.activation_diff(self.V[j]) * (self.w[j].T @ self.local_gradient[j+1])
            self.delta_w[j-1] = (self.momentum_rate * self.delta_w[j-1]) + np.outer(self.learning_rate * self.local_gradient[j], self.V[j-1])
            self.delta_bias[j-1] = (self.momentum_rate * self.delta_bias[j-1]) + self.learning_rate * self.local_gradient[j]
            self.w[j-1] += self.delta_w[j-1]
            self.b[j-1] += self.delta_bias[j-1]
        return np.sum(error**2) / 2

    
    def train(self, input, design_output, Epoch = 10000, L_error = 0.001):
        N = 0
        keep_error = []
        er = 10000
        while N < Epoch and er > L_error:
            actual_output = []
            er = 0
            for i in range(len(input)):
                self.feed_forward(input[i])
                actual_output.append(self.V[-1])
                er += self.back_propagation(design_output[i])
            er /= len(input)
            keep_error.append(er)
            N += 1
            print(f"Epoch = {N} | AV_Error = {er}")
        plt.subplot(2, 1, 1)
        plt.plot(keep_error)
        plt.title('MSE vs. Epoch of TrainSet')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        # plt.show()

    def test(self, input, design_output, type = "classification"):
        actual_output = []
        for i in input:
            self.feed_forward(i)
            actual_output.append(self.V[-1])
        
        if type == "classification":
            for i in range(len(actual_output)):
                actual_output[i] = 0 if actual_output[i][0] > actual_output[i][1] else 1
                design_output[i] = 0 if design_output[i][0] > design_output[i][1] else 1
            kk = 0
            for i in range(len(actual_output)):
                if actual_output[i] == design_output[i]:
                    kk += 1
            # print(f"Accuracy = {kk*100/len(actual_output)}%")
        else:
            actual_output = [element[0] for element in actual_output]
        categories = [f"{element}" for element in range(len(design_output))]
        er = 0
        for i in range(len(actual_output)):
            er += np.sum((actual_output[i]-design_output[i])**2) / 2
        er /= len(actual_output)
        bar_width = 0.2
        index = range(len(categories))
        plt.subplot(2, 1, 2)
        plt.bar(index, np.array(actual_output), bar_width, label='Actual Output', color='b')
        plt.bar([i + bar_width for i in index], np.array(design_output), bar_width, label='Design Output', color='orange')
        plt.bar([j + bar_width for j in [i + bar_width for i in index]], np.array([100 if abs(actual_output[i]-design_output[i])>100 else abs(actual_output[i]-design_output[i]) for i in range(len(actual_output))]), bar_width, label='Error', color='r')
        plt.xlabel('Sample')
        plt.ylabel('Output')
        plt.title('Actual Output vs. Design Output of TestSet')
        plt.xticks([i + bar_width / 2 for i in index], categories)
        plt.legend()
                    


def Read_Data(data_type = "regression"):
    if data_type == "regression":
        return Read_Data1()
    elif data_type == "classification":
        return Read_Data2()

def Read_Data1(filename = 'Flood_Data.txt'):
    data = []
    input = []
    design_output = []
    with open(filename) as f:
        for line in f.readlines()[2:]:
            data.append([float(element[:-1]) for element in line.split()])
    data = np.array(data)
    np.random.shuffle(data)
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    epsilon = 1e-6  # A very small value to avoid division by zero
    data = (data - min_vals) / (max_vals - min_vals + epsilon)
    for i in data:
        input.append(i[:-1])
        design_output.append(np.array(i[-1]))
    return input, design_output

def Read_Data2(filename = 'cross.txt'):
    data = []
    input = []
    design_output = []
    with open(filename) as f:
        a = f.readlines()
        for line in range(1, len(a), 3):
            z = np.array([float(element) for element in a[line][:-1].split()])
            zz = np.array([float(element) for element in a[line+1].split()])
            data.append(np.append(z, zz))
    data = np.array(data)
    np.random.shuffle(data)
    for i in data:
        input.append(i[:-2])
        design_output.append(i[-2:])
    return input, design_output

def k_fold_varidation(data, k = 10):
    fold_size = len(data) // k
    test = []
    train = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else len(data)  # เผื่อส่วนสุดท้ายรวมทั้งหมด
        test_fold = data[start:end]
        train_fold = data[:start] + data[end:]
        test.append(test_fold)
        train.append(train_fold)
    return train, test

if __name__ == "__main__":
    
##------------------------------------ สำหรับแก้ไขค่าต่างๆ ------------------------------------##
    k = 10 # กำหนด k-fold-varidation
    layer = {"regression":[8,16,1], "classification":[2,16,2]}  # [input, hidden, output] โดย hidden สร้างได้หลาย layer เช่น [2,16,16,2]
    learning_rate = 0.3
    momentum_rate = 0.8
    Max_Epoch = 1000
    AV_error = 0.001
    activation_function = 'sigmoid' # activation_function = 'sigmoid' or 'relu' or 'tanh' or 'linear'
    data_type = "classification" # data_type = "regression" or "classification
##-----------------------------------------------------------------------------------------##

    if input("Do you want to change the default value? (y/n): ").lower() == "n":
        in_k = input("Enter k-fold-varidation: ")
        k = int(in_k) if in_k != "" else k
        learning_rate_k = input("Enter learning rate: ")
        learning_rate = float(learning_rate_k) if learning_rate_k != "" else learning_rate
        momentum_rate_k = input("Enter momentum rate: ")
        momentum_rate = float(momentum_rate_k) if momentum_rate_k != "" else momentum_rate
        Max_Epoch_k = input("Enter Max Epoch: ")
        Max_Epoch = int(Max_Epoch_k) if Max_Epoch_k != "" else Max_Epoch
        AV_error_k = input("Enter AV error: ")
        AV_error = float(AV_error_k) if AV_error_k != "" else AV_error
        data_type = input("Enter data type: ")
        if data_type != "regression" and data_type != "classification":
            data_type = "classification"

    print(f"k-fold-varidation = {k}")
    print(f"learning rate = {learning_rate}")
    print(f"momentum rate = {momentum_rate}")
    print(f"Max Epoch = {Max_Epoch}")
    print(f"AV error = {AV_error}")
    print(f"data type = {data_type}")
    
        
    #  ทำ k-fold
    input, design_output = Read_Data(data_type)
    input_train, input_test = k_fold_varidation(input, k)
    design_output_train, design_output_test = k_fold_varidation(design_output, k)

    # สร้างโมเดลตั้งต้น
    nn = NN(layer[data_type], learning_rate, momentum_rate, activation_function) 

    # ทดสอบโมเดลแบบ cross validation
    for i in range(len(input_train)):
        plt.figure(i+1)
        nn_copy = copy.deepcopy(nn)
        nn_copy.train(input_train[i], design_output_train[i], Epoch=Max_Epoch, L_error=AV_error)
        nn_copy.test(input_test[i], design_output_test[i], type=data_type)
    plt.show()    