
## This is the k-nearest neighbors workbook for ECE 239AS Assignment #2

Please follow the notebook linearly to implement k-nearest neighbors.

Please print out the workbook entirely when completed.

We thank Serena Yeung & Justin Johnson for permission to use code written for the CS 231n class (cs231n.stanford.edu).  These are the functions in the cs231n folders and code in the jupyer notebook to preprocess and show the images.  The classifiers used are based off of code prepared for CS 231n as well.

The goal of this workbook is to give you experience with the data, training and evaluating a simple classifier, k-fold cross validation, and as a Python refresher.

## Import the appropriate libraries


```python
import numpy as np # for doing most of our calculations
import matplotlib.pyplot as plt# for plotting
from cs231n.data_utils import load_CIFAR10 # function to load the CIFAR-10 dataset.

# Load matplotlib images inline
%matplotlib inline

# These are important for reloading any code you write in external .py files.
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
%load_ext autoreload
%autoreload 2
```


```python
# Set the path to the CIFAR-10 data
cifar10_dir = 'cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
```

    Training data shape:  (50000, 32, 32, 3)
    Training labels shape:  (50000,)
    Test data shape:  (10000, 32, 32, 3)
    Test labels shape:  (10000,)



```python
# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()
```


![png](output_4_0.png)



```python
# Subsample the data for more efficient code execution in this exercise
num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)
```

    (5000, 3072) (500, 3072)


# K-nearest neighbors

In the following cells, you will build a KNN classifier and choose hyperparameters via k-fold cross-validation.


```python
# Import the KNN class

from nndl import KNN
```


```python
# Declare an instance of the knn class.
knn = KNN()

# Train the classifier.
#   We have implemented the training of the KNN classifier.
#   Look at the train function in the KNN class to see what this does.
knn.train(X=X_train, y=y_train)
```

## Questions

(1) Describe what is going on in the function knn.train().

(2) What are the pros and cons of this training step?

## Answers

(1) We just memorize the data, i.e. store the data in class variables.

(2) The pros are that training is fast and simple, because it's just a simple variable assignment. The cons are that it requires a large memory overhead, i.e. the bigger the dataset the more memory we use. For datasets that don't fit in memory this won't really work

## KNN prediction

In the following sections, you will implement the functions to calculate the distances of test points to training points, and from this information, predict the class of the KNN.


```python
# Implement the function compute_distances() in the KNN class.
# Do not worry about the input 'norm' for now; use the default definition of the norm
#   in the code, which is the 2-norm.
# You should only have to fill out the clearly marked sections.

import time
time_start =time.time()

dists_L2 = knn.compute_distances(X=X_test)

print('Time to run code: {}'.format(time.time()-time_start))
print('Frobenius norm of L2 distances: {}'.format(np.linalg.norm(dists_L2, 'fro')))
```

    Time to run code: 132.85851097106934
    Frobenius norm of L2 distances: 7906696.077040902


#### Really slow code

Note: 
This probably took a while. This is because we use two for loops.  We could increase the speed via vectorization, removing the for loops.

If you implemented this correctly, evaluating np.linalg.norm(dists_L2, 'fro') should return: ~7906696

### KNN vectorization

The above code took far too long to run.  If we wanted to optimize hyperparameters, it would be time-expensive.  Thus, we will speed up the code by vectorizing it, removing the for loops.


```python
# Implement the function compute_L2_distances_vectorized() in the KNN class.
# In this function, you ought to achieve the same L2 distance but WITHOUT any for loops.
# Note, this is SPECIFIC for the L2 norm.

time_start =time.time()
dists_L2_vectorized = knn.compute_L2_distances_vectorized(X=X_test)
print('Time to run code: {}'.format(time.time()-time_start))
print('Difference in L2 distances between your KNN implementations (should be 0): {}'.format(np.linalg.norm(dists_L2 - dists_L2_vectorized, 'fro')))
```

    Time to run code: 1.4170951843261719
    Difference in L2 distances between your KNN implementations (should be 0): 0.0


#### Speedup

Depending on your computer speed, you should see a 10-100x speed up from vectorization.  On our computer, the vectorized form took 0.36 seconds while the naive implementation took 38.3 seconds. 

### Implementing the prediction

Now that we have functions to calculate the distances from a test point to given training points, we now implement the function that will predict the test point labels.


```python
# Implement the function predict_labels in the KNN class.
# Calculate the training error (num_incorrect / total_samples) 
#   from running knn.predict_labels with k=1

error = 1

# ================================================================ #
# YOUR CODE HERE:
#   Calculate the error rate by calling predict_labels on the test 
#   data with k = 1.  Store the error rate in the variable error.
# ================================================================ #
def get_error(distances, y,k):
    labels = knn.predict_labels(distances,k=k)
    diffs = [1 if predicted != actual else 0 for predicted, actual in zip(labels, y)]
    return sum(diffs)/len(diffs)

error = get_error(dists_L2_vectorized, y_test,k=1)
# ================================================================ #
# END YOUR CODE HERE
# ================================================================ #

print(error)
```

    0.726


If you implemented this correctly, the error should be: 0.726.

This means that the k-nearest neighbors classifier is right 27.4% of the time, which is not great, considering that chance levels are 10%.

# Optimizing KNN hyperparameters

In this section, we'll take the KNN classifier that you have constructed and perform cross-validation to choose a best value of $k$, as well as a best choice of norm.

### Create training and validation folds

First, we will create the training and validation folds for use in k-fold cross validation.


```python
# Create the dataset folds for cross-valdiation.
num_folds = 5

X_train_folds = []
y_train_folds =  []

# ================================================================ #
# YOUR CODE HERE:
#   Split the training data into num_folds (i.e., 5) folds.
#   X_train_folds is a list, where X_train_folds[i] contains the 
#      data points in fold i.
#   y_train_folds is also a list, where y_train_folds[i] contains
#      the corresponding labels for the data in X_train_folds[i]
# ================================================================ #
combined = np.zeros((X_train.shape[0], X_train.shape[1] + 1))
combined[:, :-1] = X_train
combined[:, -1] = y_train
np.random.shuffle(combined)
examples_per_fold = X_train.shape[0]//num_folds
start = 0
for fold in range(num_folds):
    X_fold, y_fold = combined[start:start + examples_per_fold, :-1], combined[start:start + examples_per_fold, -1]
    X_train_folds.append(X_fold)
    y_train_folds.append(y_fold)
    start+=examples_per_fold
    
# ================================================================ #
# END YOUR CODE HERE
# ================================================================ #

```

### Optimizing the number of nearest neighbors hyperparameter.

In this section, we select different numbers of nearest neighbors and classes which one has the lowest k-fold cross validation error.


```python
time_start =time.time()

ks = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]

# ================================================================ #
# YOUR CODE HERE:
#   Calculate the cross-validation error for each k in ks, testing
#   the trained model on each of the 5 folds.  Average these errors
#   together and make a plot of k vs. cross-validation error. Since 
#   we are assuming L2 distance here, please use the vectorized code!
#   Otherwise, you might be waiting a long time.
# ================================================================ #

k_to_errs = {}
for k in ks:
    overall_error = 0
    for i in range(num_folds):
        # leave out i for testing, and train on all of the others
        training_X = np.vstack([fold for idx, fold in enumerate(X_train_folds) if idx != i])
        training_Y = np.hstack([fold for idx, fold in enumerate(y_train_folds) if idx !=i])
        knn.train(training_X, training_Y)
        cv_X, cv_Y = np.array(X_train_folds[i]), np.array(y_train_folds[i])
        cv_distances = knn.compute_L2_distances_vectorized(cv_X)
        cv_error = get_error(cv_distances, cv_Y, k = k)
        overall_error+=cv_error
    avg_err = overall_error/num_folds
    print("got average error {} for k = {}".format(avg_err, k))
    k_to_errs[k] = avg_err

min_key = min(k_to_errs, key = k_to_errs.get)
print('lowest CV error was {} with k = {}'.format(min_key, k_to_errs[min_key]))
plt.plot(list(k_to_errs.keys()), list(k_to_errs.values()), 'bo')
plt.xlabel('K')
plt.ylabel('Cross Validation Error')
# ================================================================ #
# END YOUR CODE HERE
# ================================================================ #

print('Computation time: %.2f'%(time.time()-time_start))
```

    got average error 0.7282 for k = 1
    got average error 0.7632 for k = 2
    got average error 0.7476 for k = 3
    got average error 0.7294 for k = 5
    got average error 0.7325999999999999 for k = 7
    got average error 0.7242 for k = 10
    got average error 0.727 for k = 15
    got average error 0.7296000000000001 for k = 20
    got average error 0.7262 for k = 25
    got average error 0.7284 for k = 30
    lowest CV error was 10 with k = 0.7242
    Computation time: 73.82



![png](output_24_1.png)


## Questions:

(1) What value of $k$ is best amongst the tested $k$'s?

(2) What is the cross-validation error for this value of $k$?

## Answers:

(1) $k = 10$ was the best $k$ by cross validation error.

(2) The error was $0.7242$.

### Optimizing the norm

Next, we test three different norms (the 1, 2, and infinity norms) and see which distance metric results in the best cross-validation performance.


```python
time_start =time.time()

L1_norm = lambda x: np.linalg.norm(x, ord=1)
L2_norm = lambda x: np.linalg.norm(x, ord=2)
Linf_norm = lambda x: np.linalg.norm(x, ord= np.inf)
norms = [L1_norm, L2_norm, Linf_norm]
norm_map = {L1_norm: 'l1', L2_norm: 'l2', Linf_norm: 'linf'}
norm_vals = []
# ================================================================ #
# YOUR CODE HERE:
#   Calculate the cross-validation error for each norm in norms, testing
#   the trained model on each of the 5 folds.  Average these errors
#   together and make a plot of the norm used vs the cross-validation error
#   Use the best cross-validation k from the previous part.  
#
#   Feel free to use the compute_distances function.  We're testing just
#   three norms, but be advised that this could still take some time.
#   You're welcome to write a vectorized form of the L1- and Linf- norms
#   to speed this up, but it is not necessary.
# ================================================================ #
norm_to_errs = {}
for norm in norms:
    overall_error = 0
    for i in range(num_folds):
        # leave out i for testing, and train on all of the others
        training_X = np.vstack([fold for idx, fold in enumerate(X_train_folds) if idx != i])
        training_Y = np.hstack([fold for idx, fold in enumerate(y_train_folds) if idx !=i])
        knn.train(training_X, training_Y)
        cv_X, cv_Y = np.array(X_train_folds[i]), np.array(y_train_folds[i])
        cv_distances = knn.compute_distances(cv_X, norm = norm)
        cv_error = get_error(cv_distances, cv_Y, k = 10)
        overall_error+=cv_error
    avg_err = overall_error/num_folds
    print("got average error {} for norm = {}".format(avg_err, norm_map[norm]))
    norm_to_errs[norm] = avg_err
    norm_vals.append(avg_err)

    


min_key = min(norm_to_errs, key = norm_to_errs.get)
print("{} norm had lowest error of {}".format(norm_map[min_key], norm_to_errs[min_key]))
x_axis = np.array([1,2,3]) 
x_ticks = ['L1', 'L2', 'Linf']
plt.xticks(x_axis, x_ticks) 
plt.plot(x_axis, norm_vals, 'ro') # norm vals should be ordered as l1, l2 linf
plt.xlabel('Norm')
plt.ylabel('Cross Validation Error')

# ================================================================ #
# END YOUR CODE HERE
# ================================================================ #
print('Computation time: %.2f'%(time.time()-time_start))
```

    got average error 0.6916 for norm = l1
    got average error 0.7242 for norm = l2
    got average error 0.8326 for norm = linf
    l1 norm had lowest error of 0.6916
    Computation time: 1188.94



![png](output_28_1.png)


## Questions:

(1) What norm has the best cross-validation error?

(2) What is the cross-validation error for your given norm and k?

## Answers: 

(1) The L1 norm had the best CV error.

(2) Using k = 10 and the L1 norm, we get a cross validation error of $0.6916$.

# Evaluating the model on the testing dataset.

Now, given the optimal $k$ and norm you found in earlier parts, evaluate the testing error of the k-nearest neighbors model.


```python
error = 1

# ================================================================ #
# YOUR CODE HERE:
#   Evaluate the testing error of the k-nearest neighbors classifier
#   for your optimal hyperparameters found by 5-fold cross-validation.
# ================================================================ #

knn = KNN()
knn.train(X_train, y_train)
distances = knn.compute_distances(X_test, L1_norm)
error = get_error(distances, y_test, k = 10)

# ================================================================ #
# END YOUR CODE HERE
# ================================================================ #

print('Error rate achieved: {}'.format(error))
```

    Error rate achieved: 0.722


## Question:

How much did your error improve by cross-validation over naively choosing $k=1$ and using the L2-norm?

## Answer:

It improved by 0.004


```python

```
