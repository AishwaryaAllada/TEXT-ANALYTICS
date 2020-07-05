## 1.	Classification Accuracy Table:

| S.No             | Combinations                             |tanh    |Sigmoid|ReLU   |
| -----------------|:----------------------------------------:| ------:|------:|-------:
| 1                |No Regularizer + no dropout               |82.38%  |82.81% |82.42% |
| 2                |Regularizer in input +dropout             |79.81%  |79.04% |80.87% |
| 3                |Regularizer in hidden + dropout           |82.73%  |82.82% |82.88% |
| 4                |Regularizer in input and output + dropout |79.87%  |78.87% |80.82% |

Network is made up of *LSTM (as previous words will be remembered)* in the hidden layer with 128 nodes, followed by a *Dense layer* with 2 nodes and "softmax" activation function, with the model having the input layer as *Embedding*. “ReLU” activation function gave the best result with a regularizer in hidden layer and a dropout, when compared to “tanh” and “Sigmoid”. The accuracies were nearly the same with all the three of them. Sigmoid function is better for predicting probability as the output. For classification Tanh is the better. This can be observed in the above accuracy table. Adding dropout layer after the hidden layer, with *rate as 0.2 and regularizer(bias regularizer and embeddings regularizer with L2-norm value as 1e-2* to both the layers reduces the testing accuracy, as the model is a shallow network. L2 norm regularization gives better accuracy when its in hidden layer, compared to first or both layers. Dropout layer added models gave better accuracies, as they prevent the model from over fitting. Having L2-norm in input layer with dropout performed poorly of all.
