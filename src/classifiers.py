import numpy as np

from numpy import exp
from numpy import sum
from numpy import log

def softmax(logits, y):
    """
    Computes the loss and gradient for softmax classification.

    Args:
    - logits: A numpy array of shape (N, C)
    - y: A numpy array of shape (N,). y represents the labels corresponding to
    logits, where y[i] is the label of logits[i], and the value of y have a 
    range of 0 <= y[i] < C

    Returns (as a tuple):
    - loss: Loss scalar
    - dlogits: Loss gradient with respect to logits
    """
    loss, dlogits = None, None
    """
    TODO: Compute the softmax loss and its gradient using no explicit loops
    Store the loss in loss and the gradient in dW. If you are not careful
    here, it is easy to run into numeric instability. Don't forget the
    regularization!
    """
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################

    f_max = np.reshape(np.max(logits, axis=1), (logits.shape[0], 1))
    prob = exp(logits-f_max) / sum(exp(logits-f_max), axis=1, keepdims = True)
    
    labels = np.zeros_like(prob)
    labels[np.arange(logits.shape[0]), y] = 1.0
    loss = -sum(labels*np.log(prob)) / logits.shape[0]
    dlogits = prob - labels
    dlogits /= logits.shape[0]
    
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return loss, dlogits
