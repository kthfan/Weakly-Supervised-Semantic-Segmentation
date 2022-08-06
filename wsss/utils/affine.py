
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa 



class Affine:
    '''
    Supported affine transformations: rotate, rescale(both height and width), resize, flip, translation, shear.
    # Arguments:
        affine_weights: In random generated affine transformed images, weights of probability of different sort
                        of affine transformations.
        fill_mode:      Points outside the boundaries of the input are filled according to the given mode 
                        (one of {'constant', 'nearest', 'reflect', 'wrap'}). 
        epsilon:        Small float added to variance to avoid dividing by zero. 
        *_domain:       Domain of affine transformations arguments, i.e., domain of rotate angle.
        
    # Usage:
        I = cv2.imread("./img.jpg").astype(np.float32) / 255
        I = np.expand_dims(I, 0) # [batch_size, height, width, n_channel]
        
        affine = Affine()
        code, args = affine.random_affine_code()
        inv_args = affine.inverse_affine_code(code, args)
        
        A_I = affine.apply_affine(I, code, args)     # transformed image
        R_I = affine.apply_affine(I, code, inv_args) # inverse transformed image
    '''
    def __init__(self, affine_weights=None, fill_mode='nearest', epsilon=1e-6,
                 rotate_domain=(-np.pi/2, np.pi/2), rescale_ratio_domain=(0.5, 2),rescale_scale_domain=(1, 2),
                 rescale_x_domain=(0, 1), rescale_y_domain=(0, 1),
                 translation_x_domain=(-0.5, 0.5), translation_y_domain=(-0.5, 0.5),
                 shear_x_domain=(-0.5, 0.5), shear_y_domain=(-0.5, 0.5), shear_scale_domain=(0.5, 2)):
        
        r_flip = lambda I,v,h: self.flip(I, v>0.5, h>0.5)
        r_rescale = lambda I,r,_,y,x,s: self.rescale(I, r, 1., y, x, s)
        self.affine_functions = [self.rotate, r_rescale, self.rescale, r_flip, self.translation, self.shear]
        
        if affine_weights is None:
            affine_weights = tf.ones(len(self.affine_functions), dtype=tf.float32)
        affine_weights = tf.constant(affine_weights, dtype=tf.float32)
        affine_weights /= tf.reduce_sum(affine_weights)
        self.affine_weights = affine_weights 
        self.fill_mode = fill_mode
        self.epsilon = epsilon
        
        self._affine_weights_upper = tf.cumsum(self.affine_weights) 
        self._affine_weights_lower = tf.concat([tf.zeros(1, dtype=self.affine_weights.dtype), self._affine_weights_upper[:-1]], axis=0) 
        self._affine_weights_upper = self._affine_weights_upper + self.epsilon
        
        self._args_inverse = [
            lambda a: -a,
            lambda a: tf.stack([1/a[0], 1/a[1], a[2], a[3], 1/a[4]], axis=0),
            lambda a: tf.stack([a[0], a[1], a[2], a[3], 1/a[4]], axis=0),
            lambda a: a,
            lambda a: -a,
            lambda a: tf.stack([-a[0], -a[1], 1/a[2]/(1-a[0]*a[1]), a[3], a[4]], axis=0)
        ]
        self._args_min = tf.constant([
            [rotate_domain[0], np.nan, np.nan, np.nan, np.nan],
            [rescale_ratio_domain[0], 0, rescale_x_domain[0], rescale_y_domain[0], rescale_scale_domain[0]],
            [np.nan, np.nan, rescale_x_domain[0], rescale_y_domain[0], rescale_scale_domain[0]],
            [0, 0, np.nan, np.nan, np.nan],
            [translation_x_domain[0], translation_y_domain[0], np.nan, np.nan, np.nan],
            [shear_x_domain[0], shear_y_domain[0], shear_scale_domain[0], np.nan, np.nan]
        ], dtype=tf.float32)
        self._args_max = tf.constant([
            [rotate_domain[1], np.nan, np.nan, np.nan, np.nan],
            [rescale_ratio_domain[1], 1, rescale_x_domain[1], rescale_y_domain[1], rescale_scale_domain[1]],
            [np.nan, np.nan, rescale_x_domain[1], rescale_y_domain[1], rescale_scale_domain[1]],
            [1, 1, np.nan, np.nan, np.nan],
            [translation_x_domain[1], translation_y_domain[1], np.nan, np.nan, np.nan],
            [shear_x_domain[1], shear_y_domain[1], shear_scale_domain[1], np.nan, np.nan]
        ], dtype=tf.float32)
        
    def _inverse_args(self, code, arg):
        if code == 0:
            return self._args_inverse[0](arg)
        elif code == 1:
            return self._args_inverse[1](arg)
        elif code == 2:
            return self._args_inverse[2](arg)
        elif code == 3:
            return self._args_inverse[3](arg)
        elif code == 4:
            return self._args_inverse[4](arg)
        elif code == 5:
            return self._args_inverse[5](arg)
        else:
            # compiling or except code
            return arg
        
    def inverse_affine_code(self, code, args):
        inv_args = tf.map_fn(lambda e: self._inverse_args(e[0], e[1]) ,(code, args), dtype=args.dtype)
        return inv_args
    
    def random_affine_code(self, size):
        code = tf.expand_dims(tf.random.uniform([size]), -1)
        code = tf.where((self._affine_weights_lower<=code) & (code<self._affine_weights_upper))[:, 1]
        
        args = tf.random.uniform((size, self._args_min.shape[1]))
        a_o = tf.gather(self._args_min, code, axis=0)
        a_s =  tf.gather(self._args_max, code, axis=0) - a_o
        args = a_s*args + a_o
        return code, args
    
    def _standard_affine(self, I, code, arg):
        if code==0:
            I = self.affine_functions[0](I, arg[0])
        elif code==1:
            I = self.affine_functions[1](I, arg[0], tf.ones_like(arg[0]), arg[2], arg[3], arg[4])
        elif code==2:
            I = self.affine_functions[2](I, None, None, arg[2], arg[3], arg[4])
        elif code==3:
            I = self.affine_functions[3](I, arg[0], arg[1])
        elif code==4:
            I = self.affine_functions[4](I, arg[0], arg[1])
        elif code==5:
            I = self.affine_functions[5](I, arg[0], arg[1], arg[2])
        else:
            # compiling or except code
            pass
            
        return I
    def apply_affine(self, I, code, args):
        original_shape = I.shape
        A_I = tf.map_fn(lambda e: self._standard_affine(e[0], e[1], e[2]), (I, code, args), dtype=I.dtype)
        A_I.set_shape(original_shape) # ensure shape of images are the same after affine transformation.
        return A_I
    
    def _pad(self, I, y1, y2, x1, x2, fill_mode=None):
        if fill_mode is None: fill_mode = self.fill_mode
            
        pad = tf.stack([y1, y2, x1, x2, 0, 0])
        pad = tf.reshape(pad, (3, 2))
                        
        if fill_mode=="nearest":
            if y1>0:
                row = I[:1]
                row = tf.repeat(row, pad[0,0], axis=0)
                I = tf.concat([row, I], axis=0)
            if y2>0:
                row = I[-1:]
                row = tf.repeat(row, pad[0,1], axis=0)
                I = tf.concat([I, row], axis=0)
            if x1>0:
                col = I[:, :1]
                col = tf.repeat(col, pad[1,0], axis=1)
                I = tf.concat([col, I], axis=1)
            if x2<0:
                col = I[:, -1:]
                col = tf.repeat(col, pad[1,1], axis=1)
                I = tf.concat([I, col], axis=1)
        else:
            I = tf.pad(I, pad, mode=fill_mode)
        return I
    
    def translation(self, I, y, x, fill_mode=None):
        if fill_mode is None: fill_mode = self.fill_mode
        H, W = I.shape[0], I.shape[1]
                        
        if x>-1 and x<1:
            x = W*x
        if y>-1 and y<1:
            y = H*y
        
        x = tf.cast(x, tf.int64)
        y = tf.cast(y, tf.int64)
        
        lower = tf.reshape(tf.stack([-y, 0, -x, 0]), (2, 2))
        upper = tf.reshape(tf.stack([H-y, H, W-x, W]), (2, 2))
        lower = tf.reduce_max(lower, axis=1)
        upper = tf.reduce_min(upper, axis=1)
        
        cropped = I[lower[0]:upper[0], lower[1]:upper[1]]
        
        return self._pad(cropped, H-upper[0], lower[0], W-upper[1], lower[1], fill_mode)

    def shear(self, I, s_y=0, s_x=0, scale=1, fill_mode=None):
        if fill_mode is None: fill_mode = self.fill_mode
        A = scale*tf.stack([1, s_y, s_x, 1], axis=0)
        A = tf.reshape(A, (2, 2))
        
        h, w = I.shape[1], I.shape[2]
        C = tf.constant([[h/2], [w/2]], dtype=I.dtype)
        d = C - A @ C

        I = tfa.image.transform(I, [scale, scale*s_x, d[1, 0],
                                  scale*s_y, scale, d[0, 0],
                                  0, 0], fill_mode=fill_mode
                               )
        return I

    def flip(self, I, vertical=True, horizontal=True):
        if vertical and horizontal:
            I = I[::-1, ::-1]
        elif vertical:
            I = I[::-1]
        elif horizontal:
            I = I[:, ::-1]
        return I

    def rescale(self, I, height=None, width=None, y=None, x=None, scale=1, fill_mode=None):
        if scale >= 1:
            return self.rescale_zoomin(I, height=height, width=width, y=y, x=x, scale=scale)
        else:
            return self.rescale_zoomout(I, height=height, width=width, y=y, x=x, scale=scale, fill_mode=fill_mode)
                        
    def rescale_zoomout(self, I, height=None, width=None, y=None, x=None, scale=1, fill_mode=None):
        if fill_mode is None: fill_mode = self.fill_mode
        h, w = tf.cast(I.shape[0], I.dtype), tf.cast(I.shape[1], I.dtype)
        
        if height is None:
            height = h
        if width is None:
            width = w
        r = h / w
        ratio = height / width
        if ratio > r:
            height = h
            width = 1/ratio*h

        elif ratio < r:
            width = w
            height = ratio*w
        
        height = scale*height
        width = scale*width
        
        if y is None:
            y = (h - height) / 2
        elif y>0 and y<1:
            y = (h - height)*y
        if x is None:
            x = (w - width) / 2
        elif x>0 and x<1:
            x = (w - width)*x
        
        height = tf.cast(height, tf.int64)
        width = tf.cast(width, tf.int64)
        h = tf.cast(h, tf.int64)
        w = tf.cast(w, tf.int64)
        y = tf.cast(y, tf.int64)
        x = tf.cast(x, tf.int64)
        
        I = tf.image.resize(I, (height, width))
        
        I = self._pad(I, y, h-y-height, x, w-x-width, fill_mode=fill_mode)
        return I
        
    def rescale_zoomin(self, I, height=None, width=None, y=None, x=None, scale=1):
        h, w = tf.cast(I.shape[0], I.dtype), tf.cast(I.shape[1], I.dtype)
        if height is None:
            height = h
        if width is None:
            width = w

        r = h / w
        ratio = height / width
        if ratio > r:
            width = w
            height = ratio*w

        elif ratio < r:
            height = h
            width = 1/ratio*h
        
        height = scale*height
        width = scale*width
        if y is None:
            y = (height - h) / 2
        elif y>0 and y<1:
            y = (height-h)*y
        if x is None:
            x = (width - w) / 2
        elif x>0 and x<1:
            x = (width-w)*x
        
        height = tf.cast(height, tf.int64)
        width = tf.cast(width, tf.int64)
        h = tf.cast(h, tf.int64)
        w = tf.cast(w, tf.int64)
        y = tf.cast(y, tf.int64)
        x = tf.cast(x, tf.int64)
        
        I = tf.image.resize(I, (height, width))
        I = I[y:y+h, x:x+w]
        return I

    def rotate(self, I, theta, fill_mode=None):
        if fill_mode is None: fill_mode = self.fill_mode
        
        c, s = tf.cos(theta), tf.sin(theta)
        A = tf.stack([c, s, -s, c], axis=0)
        A = tf.reshape(A, (2, 2))
        
        h, w = I.shape[0], I.shape[1]
        C = tf.constant([[h/2], [w/2]], dtype=I.dtype)
        d = C - A @ C

        I = tfa.image.transform(I, [c, -s, d[1, 0],
                                  s, c, d[0, 0],
                                  0, 0], fill_mode=fill_mode
                               )
        return I

