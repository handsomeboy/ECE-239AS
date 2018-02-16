import numpy as np
import pdb

"""
This code was based off of code from cs231n at Stanford University, and modified for ece239as at UCLA.
"""

class KNN(object):

  def __init__(self):
    pass

  def train(self, X, y):
    """
	Inputs:
	- X is a numpy array of size (num_examples, D)
	- y is a numpy array of size (num_examples, )
    """
    self.X_train = X
    self.y_train = y

  def compute_distances(self, X, norm=None):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.
	- norm: the function with which the norm is taken.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    if norm is None:
      norm = lambda x: np.sqrt(np.sum(x**2))
      #norm = 2

    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in np.arange(num_test):
        
      for j in np.arange(num_train):
        # ================================================================ #
        # YOUR CODE HERE:
        #   Compute the distance between the ith test point and the jth       
        #   training point using norm(), and store the result in dists[i, j].     
        # ================================================================ #

        dists[i][j] = norm(X[i] - self.X_train[j])

        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #

    return dists


  def compute_L2_distances_vectorized(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train WITHOUT using any for loops.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))

    # ================================================================ #
    # YOUR CODE HERE:
    #   Compute the L2 distance between the ith test point and the jth       
    #   training point and store the result in dists[i, j].  You may 
    #    NOT use a for loop (or list comprehension).  You may only use
    #     numpy operations.
    #
    #     HINT: use broadcasting.  If you have a shape (N,1) array and
    #   a shape (M,) array, adding them together produces a shape (N, M) 
    #   array.
    # ================================================================ #

    vector_norms_X_train = np.array(np.sum(self.X_train**2, axis = 1)) # shape is 5000, column vector where each element is the norm of that feature vec
    vector_norms_X_train = vector_norms_X_train.reshape((vector_norms_X_train.shape[0], 1)) # reshape to broadcast
    vector_norms_X = np.array(np.sum(X**2, axis = 1)) # do the same thing for the input examples
    #broadcast operation: we have a (5000, 1) and a (500,)
    sums = vector_norms_X_train + vector_norms_X
    # basically, sums[i] will be a vector that is equal to vector_norms_X + vector_norms_X_train[i] (where the latter is a single element, and the sum is taken element wise on the vector)
    # so the below assert should pass
    #assert sums[0].all() == (vector_norms_X + vector_norms_X_train[0]).all()
    dists = sums.T - 2 * X.dot(self.X_train.T)
    dists = np.sqrt(dists)
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return dists


  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in np.arange(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      # ================================================================ #
      # YOUR CODE HERE:
      #   Use the distances to calculate and then store the labels of 
      #   the k-nearest neighbors to the ith test point.  The function
      #   numpy.argsort may be useful.
      #   
      #   After doing this, find the most common label of the k-nearest
      #   neighbors.  Store the predicted label of the ith training example
      #   as y_pred[i].  Break ties by choosing the smaller label.
      # ================================================================ #

      sorted_indices = list(np.argsort(dists[i]))
      k_closest_idx = sorted_indices[:k]
      label_to_occ = {}
      for idx in k_closest_idx:
        label = self.y_train[idx]
        closest_y.append(label)
        label_to_occ[label] = label_to_occ.get(label, 0) + 1
      # vote: get the most common label, break ties by index
      best_label, best_occ = None, 0
      for key, val in label_to_occ.items():
        if val > best_occ:
          best_label, best_occ = key, val
        elif val == best_occ:
          best_label = min(best_label, key)
      y_pred[i] = best_label


      # ================================================================ #
      # END YOUR CODE HERE
      # ================================================================ #

    return y_pred
