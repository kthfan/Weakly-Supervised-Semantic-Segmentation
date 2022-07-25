
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
    def __init__(self, affine_weights=None, fill_mode='nearest', 
                 rotate_domain=(-np.pi/2, np.pi/2), rescale_ratio_domain=(0.5, 2),rescale_scale_domain=(1, 2),
                 rescale_x_domain=(0, 1), rescale_y_domain=(0, 1),
                 translation_x_domain=(-0.5, 0.5), translation_y_domain=(-0.5, 0.5),
                 shear_x_domain=(-0.5, 0.5), shear_y_domain=(-0.5, 0.5), shear_scale_domain=(0.5, 2)):
        
        r_flip = lambda I,v,h: self.flip(I, v>0.5, h>0.5)
        r_rescale = lambda I,r,_,y,x,s: self.rescale(I, r, 1, y, x, s)
        self.affine_functions = [self.rotate, r_rescale, self.rescale, r_flip, self.translation, self.shear]
        self._affine_functions_argc = np.array([1, 5, 5, 2, 2, 3])
        _argc = np.array([1, 5, 5, 2, 2, 3])
        if affine_weights is None:
            affine_weights = np.ones(len(self.affine_functions), dtype=np.float64)
        affine_weights = np.array(affine_weights, dtype=np.float64)
        affine_weights /= affine_weights.sum()
        self.affine_weights = affine_weights 
        self.fill_mode = fill_mode
        
        self._args_inverse = np.array([
            lambda a: -a,
            lambda a: np.array([1/a[0], 1/a[1], a[2], a[3], 1/a[4]], dtype=a.dtype),
            lambda a: np.array([a[0], a[1], a[2], a[3], 1/a[4]], dtype=a.dtype),
            lambda a: a.copy(),
            lambda a: -a,
            lambda a: np.array([-a[0], -a[1], 1/a[2]/(1-a[0]*a[1]), a[3], a[4]], dtype=a.dtype)
        ], dtype=object)
        self._args_min = np.array([
            [rotate_domain[0], np.nan, np.nan, np.nan, np.nan],
            [rescale_ratio_domain[0], 0, rescale_x_domain[0], rescale_y_domain[0], rescale_scale_domain[0]],
            [np.nan, np.nan, rescale_x_domain[0], rescale_y_domain[0], rescale_scale_domain[0]],
            [0, 0, np.nan, np.nan, np.nan],
            [translation_x_domain[0], translation_y_domain[0], np.nan, np.nan, np.nan],
            [shear_x_domain[0], shear_y_domain[0], shear_scale_domain[0], np.nan, np.nan]
        ], dtype=np.float32)
        self._args_max = np.array([
            [rotate_domain[1], np.nan, np.nan, np.nan, np.nan],
            [rescale_ratio_domain[1], 1, rescale_x_domain[1], rescale_y_domain[1], rescale_scale_domain[1]],
            [np.nan, np.nan, rescale_x_domain[1], rescale_y_domain[1], rescale_scale_domain[1]],
            [1, 1, np.nan, np.nan, np.nan],
            [translation_x_domain[1], translation_y_domain[1], np.nan, np.nan, np.nan],
            [shear_x_domain[1], shear_y_domain[1], shear_scale_domain[1], np.nan, np.nan]
        ], dtype=np.float32)
        
    def inverse_affine_code(self, code, args):
        func = self._args_inverse[code]
        args = np.stack([f(a) for f, a in zip(func, args)], axis=0)
        return args
    
    def random_affine_code(self, size):
        code = np.random.multinomial(1, self.affine_weights, size=size)
        code = np.argmax(code, axis=1)
        args = np.random.uniform(0, 1, (size, self._args_min.shape[1]))
        a_o = self._args_min[code]
        a_s =  self._args_max[code] - a_o
        args = a_s*args + a_o
        return code, args
    
    def apply_affine(self, I, code, args):
        argc = self._affine_functions_argc[code]
        # set nan to None
        _args = args.astype(object)
        _args[np.isnan(args)] = None
        args = _args
        
        ret = []
        for i in range(len(I)):
            func = self.affine_functions[code[i]]
            a = args[i][:argc[i]]
            img = func(I[i:i+1], *a)
            ret.append(img)
        ret = tf.concat(ret, axis=0)
        return ret
    
    def _pad(self, I, y1, y2, x1, x2, fill_mode=None):
        if fill_mode is None: fill_mode = self.fill_mode
            
        pad = np.zeros((len(I.shape), 2), np.int32)

        pad[[1, 1, 2, 2], [0, 1, 0, 1]] = [y1, y2, x1, x2]

        if fill_mode=="nearest":
            if y1>0:
                row = I[:, :1]
                row = tf.repeat(row, pad[1,0], axis=1)
                I = tf.concat([row, I], axis=1)
            if y2>0:
                row = I[:, -1:]
                row = tf.repeat(row, pad[1,1], axis=1)
                I = tf.concat([I, row], axis=1)
            if x1>0:
                col = I[:, :, :1]
                col = tf.repeat(col, pad[2,0], axis=2)
                I = tf.concat([col, I], axis=2)
            if x2<0:
                col = I[:, :, -1:]
                col = tf.repeat(col, pad[2,1], axis=2)
                I = tf.concat([I, col], axis=2)
        else:
            I = tf.pad(I, pad, mode=fill_mode)
        return I
    
    def translation(self, I, y, x, fill_mode=None):
        if fill_mode is None: fill_mode = self.fill_mode
            
        H, W = I.shape[1], I.shape[2]

        if x>-1 and x<1:
            x = int(W*x)
        if y>-1 and y<1:
            y = int(H*y)
        
        y1, y2, x1, x2 = max(-y, 0), min(H-y, H), max(-x, 0), min(W-x, W)
        cropped = I[:, y1:y2, x1:x2]
        
        return self._pad(cropped, H-y2, y1, W-x2, x1, fill_mode)

    def shear(self, I, s_y=0, s_x=0, scale=1, fill_mode=None):
        if fill_mode is None: fill_mode = self.fill_mode
            
        A = scale*tf.constant([
            [1, s_y],
            [s_x, 1]
        ], dtype=tf.float32)
        h, w = I.shape[1], I.shape[2]
        C = tf.constant([[h/2], [w/2]], dtype=tf.float32)
        d = C - A @ C

        I = tfa.image.transform(I, [scale, scale*s_x, d[1, 0],
                                  scale*s_y, scale, d[0, 0],
                                  0, 0], fill_mode=fill_mode
                               )
        return I

    def flip(self, I, vertical=True, horizontal=True):
        if vertical and horizontal:
            I = I[:, ::-1, ::-1]
        elif vertical:
            I = I[:, ::-1]
        elif horizontal:
            I = I[:, :, ::-1]
        return I

    def rescale(self, I, height=None, width=None, y=None, x=None, scale=1, fill_mode=None):
        if scale >= 1:
            return self.rescale_zoomin(I, height=height, width=width, y=y, x=x, scale=scale)
        else:
            return self.rescale_zoomout(I, height=height, width=width, y=y, x=x, scale=scale, fill_mode=fill_mode)
    def rescale_zoomout(self, I, height=None, width=None, y=None, x=None, scale=1, fill_mode=None):
        if fill_mode is None: fill_mode = self.fill_mode
            
        h, w = I.shape[1], I.shape[2]
        if height is None:
            height = h
        if width is None:
            width = w
        r = h / w
        ratio = height / width
        if ratio > r:
            height = h
            width = int(1/ratio*h)

        elif ratio < r:
            width = w
            height = int(ratio*w)
        
        height = int(scale*height)
        width = int(scale*width)
        
        if y is None:
            y = (h - height) // 2
        elif y>0 and y<1:
            y = int((h - height)*y)
        if x is None:
            x = (w - width) // 2
        elif x>0 and x<1:
            x = int((w - width)*x)
        
        I = tf.image.resize(I, (height, width))
        
        I = self._pad(I, y, h-y-height, x, w-x-width, fill_mode=fill_mode)
        return I
        
    def rescale_zoomin(self, I, height=None, width=None, y=None, x=None, scale=1):
        h, w = I.shape[1], I.shape[2]
        if height is None:
            height = h
        if width is None:
            width = w

        r = h / w
        ratio = height / width
        if ratio > r:
            width = w
            height = int(ratio*w)

        elif ratio < r:
            height = h
            width = int(1/ratio*h)
        
        height = int(scale*height)
        width = int(scale*width)
        if y is None:
            y = (height - h) // 2
        elif y>0 and y<1:
            y = int((height-h)*y)
        if x is None:
            x = (width - w) // 2
        elif x>0 and x<1:
            x = int((width-w)*x)
        
        I = tf.image.resize(I, (height, width))
        I = I[:, y:y+h, x:x+w]
        return I

    def rotate(self, I, theta, fill_mode=None):
        if fill_mode is None: fill_mode = self.fill_mode
        
        c, s = np.cos(theta), np.sin(theta)
        A = tf.constant([
            [c, s],
            [-s, c]
        ], dtype=tf.float32)
        h, w = I.shape[1], I.shape[2]
        C = tf.constant([[h/2], [w/2]], dtype=tf.float32)
        d = C - A @ C

        I = tfa.image.transform(I, [c, -s, d[1, 0],
                                  s, c, d[0, 0],
                                  0, 0], fill_mode=fill_mode
                               )
        return I

