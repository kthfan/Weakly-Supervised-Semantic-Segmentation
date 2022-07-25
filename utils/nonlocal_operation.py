
import numpy as np
import tensorflow as tf

def nonlocal_operation(x, g=None, use_gaussian=True, use_relu=False, embed1=None, embed2=None, 
                       mode='spatial', epsilon=1e-6, name=""):
    '''https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.pdf
        # Arguments:
            x: Input feature map.
            g: Function or tensor, the interested feature map.
            use_gaussian: Whether to apply softmax on correlation matrix.
            use_relu: Whether to apply relu on correlation matrix.
            embed1: Embedded feature map for building the correlation matrix.
            embed2: Embedded feature map for building the correlation matrix.
            mode: "spatial" or "channel", apply correlation attention on channel or spatial information.
            epsilon: Small float added to variance to avoid dividing by zero. 
            name: String name of the layers, can not be left empty. 
    '''
    if embed1 is None:
        embed1 = x
    if embed2 is None:
        embed2 = x
    if g is None:
        g = x
    
    x_i = embed1 if isinstance(embed1, (np.ndarray, tf.Tensor)) or tf.keras.backend.is_keras_tensor(embed1) else embed1(x)
    x_j = embed2 if isinstance(embed2, (np.ndarray, tf.Tensor)) or tf.keras.backend.is_keras_tensor(embed2) else embed2(x)
    gx = g if isinstance(g, (np.ndarray, tf.Tensor)) or tf.keras.backend.is_keras_tensor(g) else g(x)
    
    x_i = tf.keras.layers.Reshape((x_i.shape[1]*x_i.shape[2], x_i.shape[3]), name=name+"_x_i")(x_i)
    x_j = tf.keras.layers.Reshape((x_j.shape[1]*x_j.shape[2], x_j.shape[3]), name=name+"_x_j")(x_j)
    
    if mode == 'spatial':
        x_j = tf.transpose(x_j, (0, 2, 1)) 
    elif mode == 'channel':
        x_i = tf.transpose(x_j, (0, 2, 1)) 
        
    corr_mat = x_i @ x_j
    if use_gaussian:
        corr_mat = tf.nn.softmax(corr_mat, axis=2)
    else:
        n_i = tf.linalg.norm(x_i, axis=2, keepdims=True)
        n_j = tf.linalg.norm(x_j, axis=1, keepdims=True)
        n_mat = n_i @ n_j
        corr_mat = corr_mat / (n_mat + epsilon)
        if use_relu:
            corr_mat = tf.nn.relu(corr_mat)
        corr_mat = corr_mat / (tf.reduce_sum(corr_mat, axis=2, keepdims=True) + epsilon)
    
    gx = tf.keras.layers.Reshape((gx.shape[1]*gx.shape[2], gx.shape[3]), name=name+"_g")(gx)
    if mode == 'channel':
        gx = tf.transpose(gx, (0, 2, 1)) 
        
    f = corr_mat @ gx
    f = tf.keras.layers.Reshape((x.shape[1], x.shape[2], -1), name=name+"_f")(f)
    return f