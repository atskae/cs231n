from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        coeff_class = 0
        for j in range(num_classes):
            if j == y[i]:
                continue

            margin = scores[j] - correct_class_score + 1  # note delta = 1
            coeff = 0
            if margin > 0:
                loss += margin
                coeff = 1

            # Compute gradient of the incorrect class
            dW[:, j] += coeff * X[i]

            # Keep track of sum to use for the correct class
            coeff_class += coeff

        # Compute the gradient for the correct class
        dW[:, y[i]] += -1 * coeff_class * X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    # The derivative of the loss needs to scale by the same amount (1/num_train)
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    # Loss function with regularization is now: loss + reg * np.sum(W*W)
    # np.sum(W*W = W^2): square each value in W, then sum up the values
    # np.sum(W^2) = w_00^2 + w_01^2 + ... + w_ij^2
    # Derivative of np.sum(W^2) = 2*w_00 + 2*w_01 + ... + 2*w_ij
    #   Pull out the 2 constant: 2(w_00 + w_01 + ... + w_ij)
    #   = 2*W
    # Derivative = dW + reg * 2*w (reg is a constant)
    dW = dW + reg * 2 * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    num_examples = X.shape[0]  # X.shape = (num_examples, num_pixels)
    num_classes = W.shape[1]  # W.shape = (num_pixels, num_classes)
    print("num_examples", num_examples)
    print("num_classes", num_classes)
    print("y.shape", y.shape)

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Matrix of scores, where each row contains the class scores for
    # that image example
    scores = np.dot(X, W)
    assert scores.shape == (num_examples, num_classes)

    correct_class_scores = np.take(scores, y, axis=1)
    correct_class_scores = scores[np.arange(scores.shape[0]), y]
    assert correct_class_scores.shape == (num_examples,)
    correct_class_scores = np.reshape(correct_class_scores, (num_examples, 1))
    print("scores.shape", scores.shape)
    print("correct_class_scores.shape", correct_class_scores.shape)

    margins = np.maximum(0, scores - correct_class_scores + 1)
    assert margins.shape == scores.shape

    # Print first example
    example_num = 25
    print("Example %i" % example_num)
    print("scores[0]: ", scores[0])  # row of scores for first example
    print("scores: ", scores[example_num])  # row of scores for this example
    print("correct class: ", y[example_num])
    print("correct_class_scores: ", correct_class_scores[example_num])
    assert scores[example_num][y[example_num]] == correct_class_scores[example_num]

    # Zero-out the score of the correct class using fancy indexing
    margins[np.arange(margins.shape[0]), y] = 0
    assert margins[example_num][y[example_num]] == 0

    # Compute the final loss by adding up the margins
    # Sum across the rows
    loss = np.sum(margins, axis=1)
    loss = np.sum(loss, axis=0) / num_examples
    print("loss", loss)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
