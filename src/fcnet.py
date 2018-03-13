import numpy as np
import os

from numpy.random import normal

from src.classifiers import softmax
from src.layers import (linear_forward, linear_backward, relu_forward,
                        relu_backward, dropout_forward, dropout_backward)

from pickle import dump

def random_init(n_in, n_out, weight_scale=5e-2, dtype=np.float32):
    """
    Weights should be initialized from a normal distribution with standard
    deviation equal to weight_scale and biases should be initialized to zero.

    Args:
    - n_in: The number of input nodes into each output.
    - n_out: The number of output nodes for each input.
    """
    W = None
    b = None
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################
    W = (normal(0.0, weight_scale, size=(n_in, n_out))).astype(dtype)
    b = np.zeros(n_out, dtype=dtype)

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return W, b



class FullyConnectedNet(object):
    """
    Implements a fully-connected neural networks with arbitrary size of
    hidden layers. For a network with N hidden layers, the architecture would
    be as follows:
    [linear - relu - (dropout)] x (N - 1) - linear - softmax
    The learnable params are stored in the self.params dictionary and are
    learned with the Solver.
    """
    def __init__(self, hidden_dims, input_dim=32*32*3, num_classes=10,
                 dropout=0, reg=0.0, weight_scale=1e-2, dtype=np.float32,
                 seed=None):
        """
        Initialise the fully-connected neural networks.
        Args:
        - hidden_dims: A list of the size of each hidden layer
        - input_dim: A list giving the size of the input
        - num_classes: Number of classes to classify.
        - dropout: A scalar between 0. and 1. determining the dropout factor.
        If dropout = 0., then dropout is not applied.
        - reg: Regularisation factor.

        """
        self.dtype = dtype
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.use_dropout = True if dropout > 0.0 else False
        if seed:
            np.random.seed(seed)
        self.params = dict()

        self.weight_scale = weight_scale
        self.hidden_dims = hidden_dims
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        """
        TODO: Initialise the weights and bias for all layers and store all in
        self.params. Store the weights and bias of the first layer in keys
        W1 and b1, the weights and bias of the second layer in W2 and b2, etc.
        Weights and bias are to be initialised according to the Xavier
        initialisation (see manual).
        """
        #######################################################################
        #                           BEGIN OF YOUR CODE                        #
        #######################################################################
        # Initialize hidden layers
        self.init_params()

        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################
        # When using dropout we need to pass a dropout_param dictionary to
        # each dropout layer so that the layer knows the dropout probability
        # and the mode (train / test). You can pass the same dropout_param to
        # each dropout layer.
        self.dropout_params = dict()
        if self.use_dropout:
            self.dropout_params = {"train": True, "p": dropout}
            self.dropout_params["seed"] = seed
                
        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

        self.print_params()

    def init_params(self):
        for idx in range(self.num_layers):
            n_in = self.input_dim if idx==0 else self.hidden_dims[idx-1]
            n_out = self.hidden_dims[idx] if idx!=len(self.hidden_dims) else self.num_classes
            W, b = random_init(n_in, n_out, self.weight_scale, self.dtype)

            self.params["W"+str(idx+1)] = W
            self.params["b"+str(idx+1)] = b

    def set_hidden_dims(self, hidden_dims):
        self.hidden_dims = hidden_dims
        self.num_layers = 1 + len(hidden_dims)
        self.reset()
            
    def reset(self):
        self.params = dict()
        self.init_params()
        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)
        
    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.
        Args:
        - X: Input data, numpy array of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].
        Returns:
        If y is None, then run a test-time forward pass of the model and
        return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.
        If y is not None, then run a training-time forward and backward pass
        and return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping
        parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        X = X.astype(self.dtype)
        linear_cache = dict()
        relu_cache = dict()
        dropout_cache = dict()
        """
        TODO: Implement the forward pass for the fully-connected neural
        network, compute the scores and store them in the scores variable.
        """
        #######################################################################
        #                           BEGIN OF YOUR CODE                        #
        #######################################################################
        train = True if y is not None else False
        
        if self.use_dropout:
            p = self.dropout_params["p"]
            seed = self.dropout_params["seed"]
            
        for layer in range(1, self.num_layers+1):
            W = self.params["W"+str(layer)]
            b = self.params["b"+str(layer)]
            input = X if layer==1 else (dropout_cache[layer-1][0] if self.use_dropout else relu_cache[layer-1])
            if layer != self.num_layers:
                linear_cache[layer] = linear_forward(input, W, b)
                relu_cache[layer] = relu_forward(linear_cache[layer])
                if self.use_dropout:
                    dropout_cache[layer] = dropout_forward(relu_cache[layer], p, train, seed)
            else:
                scores = linear_forward(input, W, b)
        
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################
        # If y is None then we are in test mode so just return scores
        if not train:
            return scores
        loss, grads = 0, dict()

        """
        TODO: Implement the backward pass for the fully-connected net. Store
        the loss in the loss variable and all gradients in the grads
        dictionary. Compute the loss with softmax. grads[k] has the gradients
        for self.params[k]. Add L2 regularisation to the loss function.
        NOTE: To ensure that your implementation matches ours and you pass the
        automated tests, make sure that your L2 regularization includes a
        factor of 0.5 to simplify the expression for the gradient.
        """
        #######################################################################
        #                           BEGIN OF YOUR CODE                        #
        #######################################################################
        loss, dscores = softmax(scores, y)
        loss += self.l2regular()

        for layer in reversed(range(1, self.num_layers+1)):
            w_key = "W" + str(layer)
            b_key = "b" + str(layer)
            W = self.params[w_key]
            b = self.params[b_key]
            if layer == self.num_layers:
                input_linear = dropout_cache[layer-1][0] if self.use_dropout else relu_cache[layer-1]
                dX, dW, db = linear_backward(dscores, input_linear, W, b)
                dW += self.reg * W
            else:
                if self.use_dropout:
                    input_drop = relu_cache[layer]
                    mask = dropout_cache[layer][1]
                    dX = dropout_backward(dX, mask, p, train)
                input_relu = linear_cache[layer]
                dX = relu_backward(dX, input_relu)
                if layer == 1:
                    input_linear = X
                else:
                    input_linear = dropout_cache[layer-1][0] if self.use_dropout else relu_cache[layer-1]
                dout = dX
                dX, dW, db = linear_backward(dout, input_linear, W, b)
                dW += self.reg * W
            grads[w_key] = dW
            grads[b_key] = db
        
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################
        return loss, grads

    def l2regular(self):
        reg = 0.0
        for layer in range(1, self.num_layers+1):
            W = self.params["W"+str(layer)]
            reg += np.sum(W**2)
        reg *= 0.5*self.reg
        return reg

    def predict(self, X):
        scores = self.loss(X)
        return np.argmax(scores, axis=1)

    def save(self, name=None):
        path = "models/"
        if not os.path.exists(path):
            os.makedirs(path)
        
        name = "fcnn.model" if name is None else name+"_fcnn.model"
        with open(path + name, 'wb') as file:
            dump(self, file)

    def print_params(self):
        params = {
            'dropout' : self.dropout_params['p'] if self.use_dropout else 0,
            'seed' : self.dropout_params['seed'] if self.use_dropout else 0,
            'reg' : self.reg,
            'weight_sacle' : self.weight_scale,
        }

        print ("#####################################################################")
        for key, value in params.items():
            stat = "{:15} : {}".format(key, value)
            print (stat)
        print ("#####################################################################")
        
