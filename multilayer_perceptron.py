import numpy as np
import matplotlib.pyplot as plt

# Activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.power(x, 2)

# Loss function
def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true.argmax(axis=1)])
    loss = np.sum(log_likelihood) / m
    return loss

# MLP class implementation
class MLP:
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu', output_activation='softmax'):
        """
        Initialize the MLP with given architecture
        
        Parameters:
        - input_size: number of input features
        - hidden_sizes: list of integers specifying number of neurons in each hidden layer
        - output_size: number of output neurons
        - activation: activation function for hidden layers ('relu', 'sigmoid', 'tanh')
        - output_activation: activation for output layer ('softmax' for classification)
        """
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.activation = activation
        self.output_activation = output_activation
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(len(self.layer_sizes)-1):
            # He initialization for ReLU, Xavier for sigmoid/tanh
            if activation == 'relu':
                std = np.sqrt(2. / self.layer_sizes[i])
            else:
                std = np.sqrt(1. / self.layer_sizes[i])
                
            self.weights.append(np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * std)
            self.biases.append(np.zeros((1, self.layer_sizes[i+1])))
    
    def forward(self, X):
        """Forward propagation through the network"""
        self.activations = [X]
        self.z_values = []
        
        # Hidden layers
        for i in range(len(self.weights)-1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            if self.activation == 'sigmoid':
                a = sigmoid(z)
            elif self.activation == 'tanh':
                a = tanh(z)
            else:  # ReLU
                a = relu(z)
                
            self.activations.append(a)
        
        # Output layer
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        
        if self.output_activation == 'softmax':
            a = softmax(z)
        elif self.output_activation == 'sigmoid':
            a = sigmoid(z)
        else:
            a = z  # linear activation
            
        self.activations.append(a)
        
        return self.activations[-1]
    
    def backward(self, X, y, learning_rate):
        """Backward propagation and parameter updates"""
        m = X.shape[0]
        deltas = [None] * len(self.weights)
        
        # Output layer error
        if self.output_activation == 'softmax':
            # For softmax with cross-entropy, the gradient simplifies to (y_pred - y_true)
            deltas[-1] = (self.activations[-1] - y) / m
        else:
            if self.output_activation == 'sigmoid':
                error = self.activations[-1] - y
                delta = error * sigmoid_derivative(self.activations[-1])
            else:  # linear
                delta = self.activations[-1] - y
            deltas[-1] = delta
        
        # Backpropagate through hidden layers
        for i in range(len(deltas)-2, -1, -1):
            if self.activation == 'sigmoid':
                delta = np.dot(deltas[i+1], self.weights[i+1].T) * sigmoid_derivative(self.activations[i+1])
            elif self.activation == 'tanh':
                delta = np.dot(deltas[i+1], self.weights[i+1].T) * tanh_derivative(self.activations[i+1])
            else:  # ReLU
                delta = np.dot(deltas[i+1], self.weights[i+1].T) * relu_derivative(self.activations[i+1])
            deltas[i] = delta
        
        # Update weights and biases
        for i in range(len(self.weights)):
            grad_w = np.dot(self.activations[i].T, deltas[i])
            grad_b = np.sum(deltas[i], axis=0, keepdims=True)
            
            self.weights[i] -= learning_rate * grad_w
            self.biases[i] -= learning_rate * grad_b
    
    def train(self, X, y, epochs, learning_rate, batch_size=32, verbose=True):
        """Train the network using mini-batch gradient descent"""
        losses = []
        accuracies = []
        
        for epoch in range(epochs):
            # Shuffle the data
            permutation = np.random.permutation(X.shape[0])
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]
            
            epoch_loss = 0
            correct = 0
            total = 0
            
            # Mini-batch training
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                output = self.forward(X_batch)
                
                # Calculate loss
                loss = cross_entropy_loss(y_batch, output)
                epoch_loss += loss * X_batch.shape[0]
                
                # Calculate accuracy
                predictions = np.argmax(output, axis=1)
                true_labels = np.argmax(y_batch, axis=1)
                correct += np.sum(predictions == true_labels)
                total += X_batch.shape[0]
                
                # Backward pass and update
                self.backward(X_batch, y_batch, learning_rate)
            
            # Calculate epoch metrics
            epoch_loss /= X.shape[0]
            accuracy = correct / total
            losses.append(epoch_loss)
            accuracies.append(accuracy)
            
            if verbose and (epoch % 10 == 0 or epoch == epochs-1):
                print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}, Accuracy = {accuracy:.4f}")
        
        return losses, accuracies
    
    def predict(self, X):
        """Make predictions using the trained network"""
        output = self.forward(X)
        return np.argmax(output, axis=1)
    
    def evaluate(self, X, y):
        """Evaluate the model on test data"""
        output = self.forward(X)
        predictions = np.argmax(output, axis=1)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == true_labels)
        loss = cross_entropy_loss(y, output)
        return loss, accuracy

# Data preprocessing functions
def train_test_split(X, y, test_size=0.2, random_state=None):
    """Split data into training and test sets"""
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    n_test = int(test_size * n_samples)
    
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

def normalize(X_train, X_test):
    """Normalize features using training set statistics"""
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    
    return X_train_norm, X_test_norm

def one_hot_encode(y, num_classes):
    """Convert class labels to one-hot encoding"""
    return np.eye(num_classes)[y]

# Example usage with Iris dataset
def load_iris_data():
    from sklearn.datasets import load_iris
    data = load_iris()
    X = data.data
    y = data.target
    return X, y

def main():
    # Load and preprocess data
    X, y = load_iris_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test = normalize(X_train, X_test)
    
    # Convert labels to one-hot encoding
    y_train = one_hot_encode(y_train, 3)
    y_test = one_hot_encode(y_test, 3)
    
    # Create and train MLP
    mlp = MLP(input_size=4, hidden_sizes=[10, 8], output_size=3, activation='relu')
    losses, accuracies = mlp.train(X_train, y_train, epochs=100, learning_rate=0.01, batch_size=16)
    
    # Evaluate on test set
    test_loss, test_accuracy = mlp.evaluate(X_test, y_test)
    print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.show()
def print_dataset():
    import pandas as pd
    from sklearn.datasets import load_iris

    # Load the dataset
    iris = load_iris()

    # Convert to pandas DataFrame
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

    # Add target column (species)
    df['species'] = iris.target
    df['species'] = df['species'].map({i: name for i, name in enumerate(iris.target_names)})

# Display the first 10 rows
    print(df.head(100))


if __name__ == "__main__":
    main()
    from sklearn.datasets import load_iris
    # print(load_iris())
    # print_dataset()

    data = load_iris()
    print(data.target_names)
