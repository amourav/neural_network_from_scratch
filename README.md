# neural_network_from_scratch

Python neural network implementation with vectorized forward and back propagation.

Comparison with Keras on classification (`neural_network_demo_classification.ipynb`) and regression (`neural_network_demo_regression.ipynb`) tasks.

## Description

Artificial neural networks (ANN) systems are computing systems that are inspired by biological neural networks. Such systems "learn" to perform tasks by considering examples, generally without being programmed with task-specific rules. For example, in image recognition, they might learn to identify images that contain cats by analyzing example images that have been manually labeled as "cat" or "no cat" and using the results to identify cats in other images. They do this without any prior knowledge of cats, for example, that they have fur, tails, whiskers and cat-like faces. Instead, they automatically generate identifying characteristics from the examples that they process.

In ANN implementations, the "signal" at a connection is a real number, and the output of each neuron is computed by some non-linear function of the sum of its inputs. The connections are called edges. Neurons and edges typically have a weight that adjusts as learning proceeds. The weight increases or decreases the strength of the signal at a connection. Neurons may have a threshold such that a signal is sent only if the aggregate signal crosses that threshold. Typically, neurons are aggregated into layers. Different layers may perform different transformations on their inputs. Signals travel from the first layer (the input layer), to the last layer (the output layer), possibly after traversing the layers multiple times.

![Image](imgs\neural_network.png) <br/>
ANN illustration.

source: https://en.wikipedia.org/wiki/Artificial_neural_network

## Dataset
Here I tried to replicate the performance of the identical network architecture implemented in Keras on some toy datasets.

#### Classification
Iris Dataset (https://en.wikipedia.org/wiki/Iris_flower_data_set)
![Image](imgs\clf_model_hist.png) <br/>
![Image](imgs\clf_comparison.png) <br/>
andreiNet Accuracy:
```
trn acc 1.0
test acc 0.908
```
Keras Accuracy:
```
trn acc 1.0
test acc 0.908
```

#### Regression
Boston House Prices Dataset (http://lib.stat.cmu.edu/datasets/boston)
![Image](imgs\reg_model_hist.png) <br/>
![Image](imgs\reg_comparison.png) <br/>
andreiNet MSE:
```
trn MSE 3.74
test MSE 10.72
```
Keras MSE:
```
trn MSE 3.55
test MSE 11.82
```

## Dependencies

The only dependency for andreiNet is `numpy`.
To run the demo notebooks you will need a few additional packages:
`matplotlib`
`sklearn`
`Keras`
`tensorflow`


## Usage

`nn = NeuralNetwork()` Initialize neural network.

`nn.fit(X_trn, y_trn)` Train neural network.

`y_pred = nn.predict(X_test)` Run inference on data.

example:

```
from andreiNet.neural_net import NeuralNetwork
from andreiNet.utils import norm_data, one_hot_encode
from andreiNet.metrics import accuracy
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Load Dataset
iris = datasets.load_iris()
X = iris.data  
y = iris.target

# We will also split the dataset into training and testing so we can evaluate the kNN classifier
X_trn, X_test, y_trn, y_test = train_test_split(X, 
                                                  y, 
                                                  test_size=0.80, 
                                                  random_state=0,
                                                  stratify=y)

X_trn_norm, (trn_mean, trn_std) = norm_data(X_trn)
X_test_norm = (X_test - trn_mean) / trn_std                                             y, 
                                                 test_size=0.333, 
                                                 random_state=0,
                                                 stratify=y)
# Set parameters
activation = 'ReLU'
batch_size = 50
random_state = 0
lr = 0.001
n_epochs = 10000
loss = 'cross_entropy'
metrics = ['accuracy']
weight_init = 'he_norm'
hidden_layers = (50, 60, 50)

# Initialize model
nn = NeuralNetwork(hidden=hidden_layers, 
                   init_weights=weight_init,
                   loss=loss,
                   activation=activation,
                   shuffle=True,
                   random_state=random_state,
                   metrics=metrics,
                   verbose=False
                   )
# Train model
nn.train(X_trn_norm, y_trn, 
         n_epochs=n_epochs,
         batch_size=batch_size, 
         early_stop=None, 
         lr=lr, 
         val_data=(X_test_norm, y_test),
         save_best=True)

# Run Inference
y_pred_trn = nn.predict(X_trn_norm).argmax(axis=1)
y_pred_test = nn.predict(X_test_norm).argmax(axis=1)

print('trn acc', accuracy(y_pred_trn, y_trn))
print('test acc', accuracy(y_pred_test, y_test))
>> epoch 10000: final trn loss = 0.0421946740950931
>> trn acc 1.0
>> test acc 0.9083333333333333
```


## Author

Andrei Mouraviev

## References

[1] https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/

[2] https://peterroelants.github.io/posts/neural-network-implementation-part04/

[3] https://stackoverflow.com/questions/57741998/vectorizing-softmax-cross-entropy-gradient
