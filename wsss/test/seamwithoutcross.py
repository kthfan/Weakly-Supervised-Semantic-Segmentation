

import numpy as np
import tensorflow as tf
from ..utils import Affine
from ..utils import nonlocal_neural_network
from ..seam import SEAM

class SEAMWithoutCross(SEAM):
    '''
    modified:
    L_ECR = tf.abs(A_F_o_r - F_t_r) + tf.abs(A_F_o - F_t)
    L_ECR += tf.abs(F_o_r - inv_A_F_t_r) + tf.abs(F_o - inv_A_F_t)

    refined-cam to refined-cam, cam to cam.
    '''
    def __init__(self, image_input, feature_output, classes, correlation_feature=None, name="seam-no-cross", **kwargs):
        super(SEAMWithoutCross, self).__init__(image_input, feature_output, classes, name=name, **kwargs)
        
        if type(self) == SEAMWithoutCross:
            outputs = self._cnn_head(feature_output, correlation_feature, classes)
            self._build_models(image_input, feature_output, *outputs)
    
    def _loss(self, affine_code, affine_args, inv_affine_args, x, y, A_x, 
              F_o, F_t, F_o_r, F_t_r, pred_y_o, pred_y_t):
            
        # only objects exist are important. (note that background always appear)
        y_bg = tf.pad(y, [[0,0],[0,1]], mode='CONSTANT', constant_values=1)
        y_bg = tf.keras.layers.Reshape((1, 1, -1))(y_bg)
        F_o, F_t, F_o_r, F_t_r = F_o*y_bg, F_t*y_bg, F_o_r*y_bg, F_t_r*y_bg
            
        # min pooling loss
        loss_min = 0
        if self.min_pooling_rate!=0:
            loss_min_o = self._min_pooling_loss(F_o_r)
            loss_min_t = self._min_pooling_loss(F_t_r)
            loss_min_o, loss_min_t = tf.reduce_mean(loss_min_o), tf.reduce_mean(loss_min_t)
            loss_min = 0.5*(loss_min_o + loss_min_t)
            
        # Note that refined cam is not normalized
        # F_o_r, F_t_r = self._max_norm(F_o_r, False), self._max_norm(F_t_r, False)
            
        # spatial equivalent regularization
        A_F_o = self._affine(F_o, affine_code, affine_args)
        A_F_o_r = self._affine(F_o_r, affine_code, affine_args)
            
        L_ER = tf.abs(A_F_o - F_t)
        L_ECR = tf.abs(A_F_o_r - F_t_r) + tf.abs(A_F_o - F_t)
            
        inv_A_F_t, inv_A_F_t_r = None, None
        if self.use_inverse_affine:
            inv_A_F_t = self._affine(F_t, affine_code, inv_affine_args)
            inv_A_F_t_r = self._affine(F_t_r, affine_code, inv_affine_args)
                
            L_ER += tf.abs(F_o - inv_A_F_t)
            L_ECR += tf.abs(F_o_r - inv_A_F_t_r) + tf.abs(F_o - inv_A_F_t)
                                                            
            L_ER, L_ECR = 0.5*L_ER, 0.5*L_ECR
            
        L_ER, L_ECR = tf.reduce_mean(L_ER, axis=(0,1,2,3)), tf.reduce_mean(L_ECR, axis=(0,1,2,3))
        L_ER, L_ECR = self.ER_reg_coef*L_ER, self.ECR_reg_coef*L_ECR
            
        # classification loss
        loss_cls_o = self.compiled_loss(y, pred_y_o, regularization_losses=None)
        loss_cls_t = self.compiled_loss(y, pred_y_t, regularization_losses=None)
        loss_cls = 0.5*(loss_cls_o + loss_cls_t)
           
        return ((pred_y_o, pred_y_t, y_bg, F_o, F_t, F_o_r, F_t_r, A_F_o, A_F_o_r, inv_A_F_t, inv_A_F_t_r),
                (loss_cls, L_ER, L_ECR, loss_min))
            