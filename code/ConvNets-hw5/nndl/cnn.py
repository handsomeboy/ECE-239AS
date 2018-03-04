import numpy as np

from nndl.layers import *
from nndl.conv_layers import *
from cs231n.fast_layers import *
from nndl.layer_utils import *
from nndl.conv_layer_utils import *

import pdb

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.use_batchnorm = use_batchnorm
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Initialize the weights and biases of a three layer CNN. To initialize:
    #     - the biases should be initialized to zeros.
    #     - the weights should be initialized to a matrix with entries
    #         drawn from a Gaussian distribution with zero mean and 
    #         standard deviation given by weight_scale.
    # ================================================================ #

    self.params['W1'], self.params['b1'] = np.random.randn(num_filters, input_dim[0], filter_size, filter_size) * weight_scale, np.zeros(num_filters)
    #hack: using pad from below
    # this just makes hout and wout input_dim[1] and 2 respectively
    pad = int((filter_size -1)/2)
    h_out, w_out = (input_dim[1] - filter_size + 2* pad) + 1, (input_dim[2] - filter_size + 2* pad) + 1
    h, w = int((h_out - 2)/2 + 1), int((w_out - 2)/2 + 1)
    # the output of the conv will be num_filters * h * w
    self.params['W2'], self.params['b2'] = np.random.randn(num_filters * h * w, hidden_dim) * weight_scale, np.zeros(hidden_dim)
    self.params['W3'], self.params['b3'] = np.random.randn(hidden_dim, num_classes) * weight_scale, np.zeros(num_classes)

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of the three layer CNN.  Store the output
    #   scores as the variable "scores".
    # ================================================================ #
    # TODO - assumes no batchnorm
    fmaps_pooled, fmaps_pooled_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    fmaps_pooled_reshaped = fmaps_pooled.reshape((fmaps_pooled.shape[0], fmaps_pooled.shape[1] * fmaps_pooled.shape[2] * fmaps_pooled.shape[3]))
    hfc_1, hfc_1_cache = affine_relu_forward(fmaps_pooled_reshaped, W2, b2)
    scores, scores_cache = affine_forward(hfc_1, W3, b3)
    

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    if y is None:
      return scores
    
    loss, grads = 0, {}

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backward pass of the three layer CNN.  Store the grads
    #   in the grads dictionary, exactly as before (i.e., the gradient of 
    #   self.params[k] will be grads[k]).  Store the loss as "loss", and
    #   don't forget to add regularization on ALL weight matrices.
    # ================================================================ #
    data_loss, data_loss_grad = softmax_loss(scores, y)
    r_loss = .5 * self.reg * (np.sum(self.params['W1'] **2) + np.sum(self.params['W2'] ** 2) + np.sum(self.params['W3'] **2))
    loss = data_loss + r_loss
    # now backwards through the network
    dx, dw_3, db_3 = affine_backward(data_loss_grad, scores_cache)
    grads['W3'] = dw_3 + self.reg * self.params['W3']
    grads['b3'] = db_3
    # now backwards with affine_relu_backwards
    dx, dw_2, db_2 = affine_relu_backward(dx, hfc_1_cache)
    grads['W2'] = dw_2 + self.reg * self.params['W2']
    grads['b2'] = db_2 
    # now backwards with conv_relu_pool_backward
    dx = dx.reshape((fmaps_pooled.shape[0], fmaps_pooled.shape[1], fmaps_pooled.shape[2], fmaps_pooled.shape[3]))
    dx, dw_1, db_1 = conv_relu_pool_backward(dx, fmaps_pooled_cache)
    grads['W1'] = dw_1 + self.reg * self.params['W1']
    grads['b1'] = db_1
    

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return loss, grads
  