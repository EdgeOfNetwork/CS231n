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
    #3073, 10
    # compute the loss and the gradient
    num_classes = W.shape[1] #10
    num_train = X.shape[0]   # 3073
    loss = 0.0

    for i in range(num_train):
        scores = X[i].dot(W) #내적 고로 score.shape은 (C,)가 된다
        correct_class_score = scores[y[i]] #y[i] = c means X[i] has label c, where 0 <= c < C.
        for j in range(num_classes):
            if j == y[i]: #j는 0~10 사이의 수 인데... j랑 y[i] 같은경우는 뭐야?
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, y[i]] -= X[i]
                print("dW 1차 : ",dW)
                dW[:, j] += X[i]
                print("dW 2차 :", dW)
                # [0,,0,0,0,0,0,0,....0] []


    """
    margin이 0보다 큰 경우에 
    correct class의 경우에는 loop 돌때 마다 input image의 pixel value 만큼 빼주고 
    
    j 번째로 loop를 돌고 있는 j 번째 class score에서는 
    input image의 pixel value만큼을 더해준다.
    """

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.

    #지금은 그냥, 로스값을 training examples의 합으로 표현했지만,
    #그러나 이 로스로 합의 평균값이 필요하다, 따라서 갯수만큼 나눴다 
    loss /= num_train
    dW /= num_train

    #1/N 구현

    # Add regularization to the loss.
    #오버피팅 방지를 위해 L2 Reg 추가
    loss += reg * np.sum(W * W)
    dW = dW + reg * 2 * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    
    #위 loss와 grad를 구하는 코드를 응용하는게 더 낫다는 얘기다.

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]

    num_train = X.shape[0]
    scores = X.dot(W)
    
    correct_class_score = scores[np.arange(num_train), y].reshape(num_train, 1)
    margin = np.maximum(0, scores - correct_class_score + 1) #1 for robustness
    
    margin[np.arange(num_train), y] = 0
    loss = margin.sum() / num_train
    loss += reg * np.sum(W * W)

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

    margin[margin > 0] = 1 #???
    valid_margin_count = margin.sum(axis = 1)

    margin[np.arange(num_train), y] -= valid_margin_count
    dW = X.T.dot(margin) / num_train

    dW = dW + reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
