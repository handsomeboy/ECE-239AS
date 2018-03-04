import numpy as np
from nndl.layers import *
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

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  pad = conv_param['pad']
  stride = conv_param['stride']


  #================================================================#
  # YOUR CODE HERE:
  #   Implement the forward pass of a convolutional neural network.
  #   Store the output as 'out'.
  #   Hint: to pad the array, you can use the function np.pad.
  # ================================================================ #

  # pad witdh 0 across the N points and C channels dims

  x_pad = np.pad(x, pad_width = ([(0,0), (0,0), (pad,pad), (pad,pad)]), mode = 'constant')
  N, _, H, W = x.shape
  F, _, HH, WW = w.shape
  h_out, w_out = int((H+2*pad-HH)/stride + 1), int((W + 2 * pad - WW)/stride + 1)
  # create the output array
  out = np.zeros((N, F, h_out, w_out))
  # pick out a window across the height and width for ALL the points across ALL the channels.
  for i in range(h_out):
    for j in range(w_out):
      # start at i * stride, and then go until that plus the desired height/width of the conv
      height_range = slice(i * stride, i * stride + HH)
      width_range = slice(j * stride, j * stride + WW)
      cur = x_pad[:,:,height_range, width_range]
      for channel, (filter, bias) in enumerate(zip(w, b)):
        convolved_result = cur * filter
        # all points, particular channel
        out[:,channel,i,j]=np.sum(convolved_result, axis = (1, 2, 3)) + bias

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #
    
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  N, F, out_height, out_width = dout.shape
  x, w, b, conv_param = cache
  
  stride, pad = [conv_param['stride'], conv_param['pad']]
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
  num_filts, _, f_height, f_width = w.shape

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the backward pass of a convolutional neural network.
  #   Calculate the gradients: dx, dw, and db.
  # ================================================================ #
  _,_,H, W = x.shape
  dx, dw, db = np.zeros(x.shape), np.zeros(w.shape), np.zeros(b.shape)
  # gradient for db is just the sum, since axis = 1 is the number of filters, we want to keep db in that shape
  db = np.sum(dout,axis=(0,2,3))
  # pad DX now so it's easier to get the windows for the subgrads, and then chop off the pads later.
  dx_pad = np.pad(dx, ((0,0), (0,0), (pad,pad), (pad,pad)), 'constant')
  # go through each example
  for i in range(N):
    # go through each channel
    for j in range(num_filts):
      for k in range(out_height):
        for l in range(out_width):
          # read off the padded window, select the current variable, all the channels, and the sliced height and width
          # and multiply it with the incoming gradient at that idx.
          height_slice, width_slice = slice(k * stride, k * stride + f_height), slice(l * stride, l * stride + f_width)
          # grad at dw is the incoming grad scaled by the values at the window
          # grad for dw is x * dout
          dw[j]+=xpad[i,:,height_slice,width_slice]*dout[i,j,k,l]
          # grad for dx is w * dout
          dx_pad[i,:,height_slice,width_slice]+=w[j] * dout[i,j,k,l]

  # remove the extra padding for dx, since we computed dx with it padded.
  return dx_pad[:,:,pad:pad+H, pad:pad+W], dw, db


  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling forward pass.
  # ================================================================ #

  # read off the shapes
  N, C, H, W = x.shape
  ph = pool_param['pool_height']
  pw = pool_param['pool_width']
  stride = pool_param['stride']
  # compute the outputs given the formula
  h_out = int((H-ph)/stride+1)
  w_out = int((W-pw)/stride+1)
  out = np.zeros((N,C,h_out,w_out))
  # take maxes across the height and width regions, meaning that we can keep the filters & num_points constant
  # ie don't have to iter across them
  for i in range(h_out):
    for j in range(w_out):
      height_slice, width_slice = slice(i * stride, i * stride + ph), slice(j * stride, j * stride + pw)
      # only take max across the height and width axes, so 2 and 3 (0 and 1) are num points and channels
      out[:,:,i,j] = np.amax(x[:,:,height_slice, width_slice], axis = (2,3))


  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 
  cache = (x, pool_param)
  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling backward pass.
  # ================================================================ #
  N, C, H, W = x.shape
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']
  hout = int(1 + (H - pool_height) / stride)
  wout = int(1 + (W - pool_height) / stride)
  dx =  np.zeros(x.shape)
  for k in range(hout):
    for l in range(wout):
      height_slice, width_slice = slice(k * stride, k * stride + pool_height), slice(l * stride, l * stride + pool_width)
      cur_region = x[:,:,height_slice, width_slice]
      # this is really ugly, but if we don't have the new axes there is a shape error
      dx[:,:,height_slice, width_slice]=(dout[:,:,k,l])[:,:,np.newaxis,np.newaxis] * (cur_region == np.amax(cur_region, axis = (2,3))[:,:,np.newaxis,np.newaxis])
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 
  return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm forward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  #Hence, one way to think of spatial batch-normalization is to reshape 
  #the (N, C, H, W) array as an (N*H*W, C) array 
  #and perform batch normalization on this array.

  # ok, so we want to make an N, C, H, W into a N,H,W,C
  # basically we need to have C as the last channel
  N, C, H, W = x.shape
  # the batch normalization is done across the examples
  # and for each examples we have the height by width
  # so we want to make 0, 1, 2, 3 -> 0, 2, 3, 1
  # the input should be batch_size * h * w * channels
  x = x.transpose(0, 2, 3, 1).reshape(N*H*W, C) # make N,C,H,W -> N, H, W, C
  # and then reshape it
  out, cache = batchnorm_forward(x, gamma, beta, bn_param)
  # the output should be batch_size * num_channels * H * w
  # i.e. the same as the input
  out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
  return out, cache


  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm backward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #

  N,C,H,W = dout.shape
  # apply the same tranposing logic again
  dout_reshaped = dout.transpose(0,2,3,1) # make N,C,H,W -> N, H, W, C
  dout_reshaped = dout_reshaped.reshape((N*H*W,C))
  dx, dgamma, dbeta = batchnorm_backward(dout_reshaped, cache)
  # the output should be batch_size * num_channels * H * w
  # i.e. the same as the input
  dx = dx.reshape(N,H,W,C).transpose(0,3,1,2)
  return dx, dgamma, dbeta
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx, dgamma, dbeta