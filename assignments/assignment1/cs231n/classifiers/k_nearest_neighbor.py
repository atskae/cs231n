from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange

# Custom packages
from math import sqrt


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

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
        for i in range(num_test):
            # Take ith test example of shape (1, 3072)
            # Image was flattened to an array of pixel values
            # of each color channel [---R--- ---G--- ---B---]
            test_example = X[i]
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

                # Take the jth training example
                # which is also a flattened image array:
                # [---R--- ---G--- ---B---]
                train_example = self.X_train[j]

                # L2 distance = Euclidean distance
                # Element-wise difference and square
                diff_squares = np.square(test_example - train_example)

                # Take the sum of all elements in array
                # np.sum() returns a scalar with axis=None
                dists[i, j] = float(sqrt(np.sum(diff_squares, axis=None)))

                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        # print('X.shape', X.shape)
        # print('X_train.shape', self.X_train.shape)
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # Array of pixels [---R---G---B---]
            # Let total pixels (R + G + B pixels) = p
            # shape = (1, p)
            test_example = X[i]

            # X.train.shape = (num_train, p)
            # Broadcasts the test example with the training examples matrix
            diff_squares = np.square(test_example - self.X_train)
            # if i == 0:
            #   print('diff_squares.shape', diff_squares.shape)
            #   print('test_example[0]', test_example)
            #   print('train_example[0]', self.X_train[0])
            #   print('diff_squares[0]', diff_squares[0][0])

            # In each row, sum across the colums
            # axis=0, sum across rows (go down columns)
            # axis=1, sum across columns (go across row)
            sm = np.sum(diff_squares, axis=1, keepdims=True)
            # if i == 0:
            #  print('sm.shape', sm.shape)
            assert sm.shape == (num_train, 1)

            temp = np.sqrt(sm)
            # if i == 0:
            #  print('temp.shape', temp.shape)

            # Transpose column vector temp to row vector
            dists[i, :] = temp.T

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Let a = X (test set) and b = X_train (training set)
        # L2 distance = sqrt( sum( (ai-bi)^2 ) )
        # (a-b)^2 = (a-b)(a-b) = a^2 - 2ab + b^2
        # = -2ab + a^2 + b^2

        # Square each element in a and b
        X_sq = np.square(X)
        X_train_sq = np.square(self.X_train)

        # print('X_sq.shape', X_sq.shape)
        # print('X_train_sq.shape', X_train_sq.shape)

        # Sum across rows of each matrix to get column vectors
        X_sm = np.sum(X_sq, axis=1, keepdims=True)
        X_train_sm = np.sum(X_train_sq, axis=1, keepdims=True)
        # print('X_sm.shape', X_sm.shape)
        # print('X_train_sm.shape', X_train_sm.shape)

        # For each element in X_sm, sum across all elements in X_train
        # X_train_sm.T becomes a row vector
        sm = X_sm + X_train_sm.T
        # print('sm.shape', sm.shape)

        dists = np.sqrt(-2 * np.dot(X, self.X_train.T) + sm)

        # X_sum = np.sum(X, axis=1, keepdims=True)
        # print('X_sum.shape', X_sum.shape)
        # X_train_sum = np.sum(self.X_train, axis=1, keepdims=True)
        # print('X_train_sum.shape', X_train_sum.shape)

        # sm = X_sum + X_train_sum.T
        # print('sm.shape', sm.shape)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.
        - k: An integer determining how many nearest neighbors to obtain

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)

        # Check k value
        num_train = dists.shape[1]
        if k < 0 or k > num_train:
            print(
                """k=%i must be non-negative integer that is <=
                the number of training examples %i"""
                % (k, num_train)
            )
            return y_pred

        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # Row/array of distances for ith test example
            # Each element in this array is the distance between
            # ith test example and the jth training example
            # Sorts dists in increasing order
            # Returns the *indices* of the sorted array
            dist_indices = np.argsort(dists[i])

            # Obtain the label of the first k training examples
            # At this point, we know k <= num_train
            for j in range(0, k):
                closest_y.append(self.y_train[dist_indices[j]])

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # Naive way
            counts = {}
            for label in closest_y:
                if label in counts:
                    counts[label] += 1
                else:
                    counts[label] = 1

            # Sort the dict insertion order to descending by value
            # Multiply by -1 for descending order
            counts = dict(sorted(counts.items(), key=lambda item: -1 * item[1]))

            # Store prediction for the ith test example
            y_pred[i] = list(counts.keys())[0]

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
