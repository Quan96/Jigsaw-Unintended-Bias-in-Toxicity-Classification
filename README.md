# Jigsaw Unintended Bias in Toxicity Classification

## Description
This was a Kaggle Competition on Natural Language Processing. The challenges was to build a model that recognizes toxicity and minimizes this type of unintended bias with respect to mentions of identities.

The dataset labeled for identity mentions and optimizing a metric designed to measure unintended bias. Develop strategies to reduce uninteded bias in machine learning models.

The data came from an archive of the [Civil Comments Platform](https://medium.com/@aja_15265/saying-goodbye-to-civil-comments-41859d3a2b1d). These public comments were create from 2015-2017 and appeared on approximately 50 English-language news sites accross the world. When Civil Comments shut down in 2017, the original data was publish on [figshare](https://figshare.com/articles/dataset/data_json/7376747).

## Code Structure and Explaination
The competition is a Kernels-only competion, which is using the computational power of the platform kernel to perform the training and predicting. It has limitation on the run-time of the kernel. As a result, I cannot make a large or too deep Recurrent Neural Network (RNN).

The code has 2 different versions, 1 uses Gated Recurrent Network (GRN) and the other uses Long-Short Term Memory (LSTM). Each version consist of following step:

1. Load embedding vectors from files
2. Build the embedding matrix
3. Build the Deep RNN model, each contains different type of layer
4. Preprocessing text, eliminate punctuation.
5. Train and Test

The problem of RNNs is Vanishing Gradients, which is happens when a network has too many layer and the gradients of the loss function go toward zero, making no update on the weights matrix even though it has not reached the optimal point.

LSTMs help prevent the vanishing gradient that can be backpropagated through time and layers by keeping a small amount of error, it allows the RNN to continue learning. The information can be stored, read or written from a cell. Its mechanism let important feature passes through learning from data. That is, cells learn when and what data allow to enter, leave or be deleted through learning.

Using the same idea and mechanism with LSTMs, a GRN, can be consider as a LSTM with less gate and without output gate. As a result, it takes less computational time and fully writes the information from its memory cell to larger net at each time step.

