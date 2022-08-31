from builtins import range
from xmlrpc.server import DocXMLRPCRequestHandler
# from socket import AF_X25
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    out = x.reshape(x.shape[0], -1)@w+b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


# 全连接W·X
def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    dx = (dout@w.T).reshape(*x.shape)
    dw = x.reshape(x.shape[0], -1).T@dout
    db = np.sum(dout ,axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


# 激活函数
def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = dout.copy()
    dx[x<=0]=0
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


# 批量正则化
def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################
        xc = x.copy()
        mean = np.mean(xc, axis=0) # 按特征计算，下同
        var = np.var(xc, axis=0)
        xc = (xc-mean)/np.sqrt(var+eps)
        out = gamma*xc+beta
        cache = {"x":x, "mean":mean, "var":var,"xc":xc,"eps":eps,"gamma":gamma,"beta":beta}

        running_mean = momentum*running_mean+(1-momentum)*mean
        running_var = momentum*running_var+(1-momentum)*var
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        out = gamma*((x-running_mean)/np.sqrt(running_var+eps))+beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables. 
    #
    ###########################################################################
    x, mean, var, xc, eps, gamma, beta = cache.values()
    N = x.shape[0]

    d_plus_1 = dout
    d_plus_2 = dout

    dbeta = np.sum(d_plus_2, axis=0)

    d_mul2_1 = d_plus_1*gamma
    d_mul2_2 = xc*d_plus_1

    dgamma = np.sum(d_mul2_2, axis=0)

    d_mul1_1 = d_mul2_1*(1/np.sqrt(var+eps))
    d_mul1_2 = d_mul2_1*(x-mean)

    d_rec = d_mul1_2*(-1/(var+eps))

    d_sqrt = d_rec*(1/(2*np.sqrt(var+eps)))

    d_sum_2 = np.sum(d_sqrt/N, axis=0)

    d_squred = d_sum_2*2*(x-mean)

    d_minus_in = d_mul1_1+d_squred # 多个分支，先汇总，其他节点因为都是（反向传播）单输入因此不需要这一步
    d_minus_1 = d_minus_in
    d_minus_2 = -d_minus_in

    d_sum_1 = np.sum(d_minus_2/N,axis=0)

    dx = d_sum_1+d_minus_1
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # cache = {"x":x, "mean":mean, "var":var,"xc":xc,"eps":eps,"gamma":gamma,"beta":beta}

    x, mean, var, xc, eps, gamma, beta = cache.values()
    N = x.shape[0]

    dgamma = np.sum(dout*xc, axis=0)
    dbeta =  np.sum(dout, axis=0)
    
    dx = np.sum(-((dout*gamma)/np.sqrt(var+eps)-np.sum((dout*gamma*(x-mean)/(2*np.sqrt(np.power(var+eps, 3))))/N, axis=0)*2*(x-mean))/N,axis=0)+\
    dout*gamma/np.sqrt(var+eps)+\
    np.sum((dout*gamma*(x-mean)*(-1/(var+eps))*(1/(2*np.sqrt(var+eps))))/N, axis=0)*2*(x-mean)

    # 理论上可以继续化简
    # dx = np.sum(-((dout*gamma)/np.sqrt(var+eps)-np.sum((dout*gamma*xc/(2*var+eps))/N, axis=0)*2*(x-mean))/N,axis=0)+\
    # dout*gamma/np.sqrt(var+eps)+\
    # np.sum((dout*gamma*xc*(-1/(var+eps))*(1/2))/N, axis=0)*2*(x-mean)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = np.random.rand(*x.shape)<p
        out = (x*mask)/p
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout.copy()
        dx[~mask] = 0
        dx[mask]/=dropout_param["p"]
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    # A naive implementation of the forward pass for a convolutional layer.

    # The input consists of N data points, each with C channels, height H and
    # width W. We convolve each input with F different filters, where each filter
    # spans all C channels and has height HH and width HH.

    # Input:
    # - x: Input data of shape (N, C, H, W)
    # - w: Filter weights of shape (F, C, HH, ww)
    # - b: Biases, of shape (F,)
    # - conv_param: A dictionary with the following keys:
    #   - 'stride': The number of pixels between adjacent receptive fields in the
    #     horizontal and vertical directions.
    #   - 'pad': The number of pixels that will be used to zero-pad the input.

    # Returns a tuple of:
    # - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    #   H' = 1 + (H + 2 * pad - HH) / stride
    #   W' = 1 + (W + 2 * pad - ww) / stride
    # - cache: (x, w, b, conv_param)
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    stride = conv_param['stride']
    pad = conv_param['pad']
    x_pad = np.pad(x,
                   ((0, 0), (0, 0), (pad, pad), (pad, pad)), # 在第0轴（N）和第1轴（C）不填充，H和W各填充pad个元素，后期需要填充，不能直接对x赋值
                   mode='constant')
    N, C, H, W = x_pad.shape
    FC, C, h, ww = w.shape
    oh = int(1+(H-h)/stride) # H'
    ow = int(1+(H-h)/stride) # W'
    out = np.zeros((N, FC, oh, ow))
    for n in np.arange(N): # 对每一个样本
      for f in np.arange(FC): # 对每一个卷积核
        cnt_i = cnt_j = 0
        w_tmp = w[f].reshape(1, -1)
        for i in np.arange(0, H-h+1, stride): # 高
            cnt_j=0
            for j in np.arange(0, W-ww+1, stride): # 宽，没设置通道是因为卷积是按照所有通道一起进行的
              x_tmp = x_pad[n, :, i:i+h, j:j+ww].reshape(1, -1)
              out[n,f,cnt_i, cnt_j] = np.sum(x_tmp*w_tmp)+b[f]
              cnt_j+=1
            cnt_i+=1

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param) # 这里返回未填充的
    return out, cache


def conv_backward_naive(dout, cache):
    """
    这种方法实际上就是计算图法，是以do为视角的方式，还有一种以dx为视角的方式，二者差不多
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
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, w, b, conv_param = cache
    stride = conv_param["stride"]
    pad = conv_param['pad']
    dx = np.zeros(x.shape)
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)
    N, C, H, W = x.shape
    FC, C, h, ww = w.shape
    oh = int(1 + (H + 2 * pad - h) / stride)
    ow = int(1 + (W + 2 * pad - ww) / stride)
    dx_paded = np.pad(dx,((0,0),(0,0),(pad,pad),(pad,pad)),'constant',constant_values = (0,0))
    x_paded = np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),'constant',constant_values = (0,0)) # x在前向传播加pad的情况下，反向传播也需要加pad
    db = np.sum(dout, axis=(0,2,3)) # 直接求和
    
    for n in range(N):
        x_i = x_paded[n] # 一个样本一个样本来
        for f in range(FC):
            w_j = w[f]
            for h_now in range(oh):
                for w_now in range(ow):
                    dw[f] += dout[n,f,h_now,w_now] * x_i[:, h_now*stride:h_now*stride+h, w_now*stride:w_now*stride+ww] # 等于X和之前导数的卷积
                    dx_paded[n, :, h_now*stride:h_now*stride+h, w_now*stride:w_now*stride+ww] += dout[n,f,h_now,w_now]*w_j
    dx = dx_paded[:, :, pad:-pad, pad:-pad] # padding值不参与反向传播
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    ###########################################################################
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
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    h = pool_param['pool_height']
    w = pool_param['pool_width']
    stride = pool_param["stride"]
    
    N, C, H, W = x.shape

    oh = int(np.ceil((H-h+1)/stride)) # H'
    ow = int(np.ceil((W-w+1)/stride)) # W'

    out = np.zeros((N, C, oh, ow))
    for n in np.arange(N): # 对每一个样本
      for c in np.arange(C):
        cnt_i = cnt_j = 0
        for i in np.arange(0, H-h+1, stride): # 高
            cnt_j=0
            for j in np.arange(0, W-w+1, stride): # 宽，没设置通道是因为卷积是按照所有通道一起进行的
              out[n, c, cnt_i, cnt_j] = np.max(x[n,c,i:i+h, j:j+w])
              cnt_j+=1
            cnt_i+=1
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    x,pool_param = cache
    h = pool_param['pool_height']
    w = pool_param['pool_width']
    stride = pool_param["stride"]
    
    N, C, H, W = x.shape
    dx = np.zeros_like(x)
    for n in np.arange(N): # 对每一个样本
      for c in np.arange(C): # 对每一个通道
        cnt_i = 0
        for i in np.arange(0, H-h+1, stride): # 高
            cnt_j=0
            for j in np.arange(0, W-w+1, stride): # 宽
              tmp_x= x[n,c,i:i+h, j:j+w].reshape(-1)
              idx = np.argmax(tmp_x)
              idx = np.eye(tmp_x.size)[idx].astype(bool)
              tmp_x[idx] = dout[n,c,cnt_i,cnt_j]
              tmp_x[~idx] = 0
              dx[n,c,i:i+h, j:j+w] = tmp_x.reshape(h, w)
              cnt_j+=1
            cnt_i+=1
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    N, C, H, W = x.shape
    a, cache = batchnorm_forward(x.transpose(0, 2, 3, 1).reshape((N * H * W, C)),gamma, beta, bn_param)
    out = a.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

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

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    N, C, H, W = dout.shape
    dbn, dgamma, dbeta = batchnorm_backward(dout.transpose(0, 2, 3, 1).reshape((N * H * W, C)), cache)
    dx = dbn.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
