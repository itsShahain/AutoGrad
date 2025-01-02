# **AutoGrad - A Reverse-mode Autodiff and Neural Network Library**

## Overview
This repository implements a **minimalistic autograd engine** and a basic **neural network library** built from scratch in Python. It serves as an educational tool to understand the core mechanics of automatic differentiation and the fundamentals of building and training neural networks.

The code is designed to be simple yet powerful, allowing users to experiment and learn without being overwhelmed by the complexity of larger libraries like TensorFlow or PyTorch.

---

## **Features**
- **Custom Automatic Differentiation (Autograd):**
  - A `Value` class that supports:
    - Forward computation of scalar operations.
    - Backpropagation to compute gradients for scalar values.
  - Support for common operations: addition, subtraction, multiplication, division, exponentiation, and activation functions like `tanh` and `exp`.

- **Custom Neural Network Components:**
  - **Neuron**: Represents a single computational unit.
  - **Layer**: A fully connected layer of neurons.
  - **MLP (Multi-Layer Perceptron)**: A feedforward neural network with multiple layers.

---

## **Installation**
Clone the repository:
```bash
git clone https://github.com/itsShahain/AutoGrad.git
cd AutoGrad
```

---

## **Usage**
Below is an example of how to use the library to define an MLP, forward a sample input, and compute gradients using backpropagation.

### 1. **Define an MLP**
Create a multi-layer perceptron with 3 input neurons and two hidden layers with 4 and 2 neurons, respectively:
```python
from neuron import MLP
from autograd import Value

# Define an MLP with input size 3 and layer sizes [4, 2]
mlp = MLP(3, [4, 2])
```

### 2. **Forward Pass**
Pass input data through the network:
```python
# Input: a list of 3 Values
inputs = [Value(0.5), Value(-1.2), Value(3.3)]
output = mlp(inputs)
print("Output:", output)
```

### 3. **Backward Pass**
Compute gradients using backpropagation:
```python
# The output of the output layer is a single Value
output[0].backward()

# Print gradients of all parameters
for param in mlp.parameters():
    print(param.grad)
```

### 4. **Accessing Parameters**
Retrieve all trainable parameters (weights and biases):
```python
parameters = mlp.parameters()
print("Number of parameters:", len(parameters))
```

---

## **Key Components**

### **1. Autograd Engine**
The `autograd.py` file implements a `Value` class that enables:
- Scalar operations (+, -, *, /, **).
- Activation functions (`tanh`, `exp`).
- Gradient computation through backpropagation.

Example:
```python
from autograd import Value

a = Value(2.0)
b = Value(3.0)
c = a * b + b ** 2
c.backward()

print("Gradient of a:", a.grad)
print("Gradient of b:", b.grad)
```

### **2. Neural Network**
The `neuron.py` file defines three classes:
- `Neuron`: A single perceptron with trainable weights and biases.
- `Layer`: A collection of neurons forming a dense layer.
- `MLP`: A multi-layer perceptron built using stacked layers.

---

## **Example: Training a Neural Network**
Here's a simple example of training an MLP using gradient descent:
```python
from neuron import MLP
from autograd import Value

# Define a simple MLP
mlp = MLP(2, [3, 1])  # 2 inputs, 1 hidden layer (3 neurons), 1 output

# Training data
inputs = [[Value(2.0), Value(3.0)], [Value(-1.0), Value(0.5)]]
targets = [Value(1.0), Value(0.0)]

# Training loop
learning_rate = 0.01
for epoch in range(100):
    total_loss = Value(0.0)
    for x, y in zip(inputs, targets):
        output = mlp(x)
        loss = (output[0] - y) ** 2  # Mean squared error
        total_loss += loss

        # Zero gradients, backward pass, and update weights
        for param in mlp.parameters():
            param.grad = 0.0
        loss.backward()

        for param in mlp.parameters():
            param.data -= learning_rate * param.grad

    print(f"Epoch {epoch + 1}, Loss: {total_loss.data}")
```
![](https://github.com/itsShahain/AutoGrad/blob/main/training.gif)

---

## **Contributing**
Contributions are welcome! If you'd like to improve this library or add new features, please:
1. Fork the repository.
2. Create a new branch for your feature.
3. Submit a pull request.

---

## **License**
MIT

---

## **Acknowledgments**
Inspired by the "micrograd" project by Andrej Karpathy, this repository aims to provide a simple, hands-on approach to understanding the basics of deep learning.
