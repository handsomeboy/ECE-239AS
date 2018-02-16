import numpy as np

if __name__ == '__main__':
	X_train = np.array([1, 2, 3]).reshape((3,1))
	X = np.array([1, 2, 3, 4, 5])
	print(X_train.shape)
	print(X.shape)
	sum = X_train + X
	print(sum.shape)
	print("X_train: {}".format(X_train))
	print("X: {}".format(X))
	print("sum: {}".format(sum))