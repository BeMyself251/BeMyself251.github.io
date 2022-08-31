import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    # 法一 原生方法*********************************************************************************
    # Y = y.copy()
    # for x,y in zip(X, Y):
    #     output = x@W
    #     exp_output = np.exp(output)
    #     softmax_output = exp_output/np.sum(exp_output) # softmax对于本样本的输出
    #     correct_output = softmax_output[y] # 此样本的正确标签对应的输出
    #     loss+=-np.log(correct_output) # 损失加上本样本的损失

    #     # i==j，因为分子存在自变量，所以和其他项的导数不同
    #     dL_da = (-1/correct_output)
    #     da_dz = correct_output*(1-correct_output)
    #     dz_dW = x
    #     dW[:, y] += dL_da*da_dz*dz_dW
        
    #     # i!=j
    #     mask = np.arange(softmax_output.size)!=y # 上方已经计算了i==j的情况，下方把所有的i!=j挑选出来
    #     dL_da = (-1/correct_output)
    #     da_dz = -(correct_output*softmax_output[mask])

    #     dz_dW = x.reshape(1, -1)
    #     dW[:, mask]+= ((dL_da*da_dz.reshape(-1, 1))@dz_dW).T
        
        
    # loss/=X.shape[0]
    # loss+=(reg*np.sum(W*W))/2
    # dW /=X.shape[0]
    # dW+=reg*W

    # 法二 数学化简后再运算*********************************************************************************
    # for i in range(X.shape[0]):
    #     # 计算前进行数学化简
    #     scores = X[i].dot(W)
    #     correct_class_score = scores[y[i]]
    #     exp_sum = np.sum(np.exp(scores))
    #     loss += np.log(exp_sum) - correct_class_score
    #     dW[:, y[i]] += - X[i]
    #     for j in range(W.shape[1]):
    #         dW[:, j] += (np.exp(scores[j]) / exp_sum) * X[i]
    # loss /= X.shape[0]
    # loss += (reg * np.sum(W * W))/2

    # dW /= X.shape[0]
    # dW += reg * W

    # 法三 one_hot trick*********************************************************************************
    Y = y.copy()
    for x,y in zip(X, Y):
        output = x@W
        exp_output = np.exp(output)
        softmax_output = exp_output/np.sum(exp_output) # softmax对于本样本的输出
        correct_output = softmax_output[y] # 此样本的正确标签对应的输出
        loss+=-np.log(correct_output) # 损失加上本样本的损失

        one_hot = np.eye(softmax_output.size)[y]
        dL_dz = (softmax_output-one_hot).reshape(1, -1)
        dz_dW = x.reshape(-1, 1)
        dW += (dL_dz.T@dz_dW.T).T # (A^T*B^T)^T = B*A 这么写是为了表明链式法则的运算顺序
        
    loss/=X.shape[0]
    loss+=(reg*np.sum(W*W))/2
    dW /=X.shape[0]
    dW+=reg*W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    batch_size = X.shape[0]
    class_size = W.shape[1]

    # 法一 原生方法*********************************************************************************
    # 前向传播
    linear_output = X@W # (batch_size, class_size) = (500, 10)
    exp_output = np.exp(linear_output) # (batch_size, class_size) = (500, 10)
    softmax_output = exp_output/np.sum(exp_output, axis=1).reshape(-1, 1) # (batch_size, class_size) = (500, 10)
    correct_output = softmax_output[np.arange(batch_size), y] # (batch_size, ) = (500, )

    # 计算损失
    loss=-np.sum(np.log(correct_output))/batch_size+0.5 * reg * np.sum(W * W)

    # 反向传播
    dL_dso = -1/(batch_size*correct_output) # (batch_size, ) = (500, )
    mask = np.arange(batch_size*class_size).reshape(batch_size, class_size) # 使用掩码获取每个样本中c==i的输出（最后一层中的一个）
    mask%=class_size
    mask = (mask==y.reshape(-1, 1)) # (batch_size, class_size) = (500, 10)，其中准确类的输出为True，其余为False

    dso_dz = np.zeros_like(softmax_output) # # (batch_size, class_size) = (500, 10)
    dso_dz[mask] = (correct_output.reshape(-1, 1)*(1-correct_output).reshape(-1, 1)).flatten() # (500, 1)*(500, 1) = (500, 1)
    dso_dz[~mask] = (-(correct_output.reshape(-1, 1)*softmax_output[~mask].reshape(batch_size, class_size-1))).flatten() # (500, 1)*(500, 9) = (500, 9).flatten() = (4500, 1)
    dL_dso = dL_dso.reshape(batch_size, -1) # (batch_size, 1) = (500, 1)
    dz_dW = X # (batch_size, pixs) = (500, pixs)

    dW = ((dL_dso*dso_dz).T@dz_dW).T
    dW += reg * W

    # 法二 数学化简后运算*********************************************************************************
    # scores = X.dot(W)
    # correct_class_score = scores[np.arange(batch_size), y].reshape(batch_size, 1)
    # exp_sum = np.sum(np.exp(scores), axis=1).reshape(batch_size, 1)
    # loss += np.sum(np.log(exp_sum) - correct_class_score)

    # # 对SoftMax的损失函数求导aL/aW
    # margin = np.exp(scores) / exp_sum
    # margin[np.arange(batch_size), y] += -1
    # dW = X.T.dot(margin)

    # # 取均值
    # loss /= batch_size
    # dW /= batch_size
    # # Add regularization to the loss.
    # # 正则化
    # loss += 0.5 * reg * np.sum(W * W)
    # dW += reg * W

    # 法三 - onehot trick*********************************************************************************
    # eye = np.eye(class_size)
    # t = eye[y]

    # y = X@W
    # y = np.exp(y)/np.sum(np.exp(y), axis=1).reshape(-1, 1)
    # dL_dz = (y-t)

    # dW = X.T@dL_dz
    # dW/=batch_size
    # dW += reg * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW