

import numpy as np
import tensorflow as tf
from ..utils import Affine
from ..utils import nonlocal_neural_network
from ..seam import SEAM

class SEAMMetricsOnly(SEAM):
    '''
    Only show L_ER, L_ECR in metrics, don't add to loss.
    '''
    def __init__(self, image_input, feature_output, classes, correlation_feature=None, name="seam-metrics-only", **kwargs):
        super(SEAMMetricsOnly, self).__init__(image_input, feature_output, classes, name=name, **kwargs)
        
        if type(self) == SEAMMetricsOnly:
            outputs = self._cnn_head(feature_output, correlation_feature, classes)
            self._build_models(image_input, feature_output, *outputs)
    
    def call(self, x, y=None, training=False):
        if training and y is not None:
            # apply affine
            affine_code, affine_args, inv_affine_args = self._ranomd_affine_code(tf.shape(x)[0])
            A_x = self._affine(x, affine_code, affine_args)
            
            # get features from simsiam model
            _, F_o, _, F_o_r, pred_y_o, _ = self.model(x, training=True)
            _, F_t, _, F_t_r, pred_y_t, _ = self.model(A_x, training=True)
        
            # compute losses
            features, losses = self._loss(affine_code, affine_args, inv_affine_args, x, y, A_x, F_o, F_t, F_o_r, F_t_r, pred_y_o, pred_y_t)
            
            loss_cls, L_ER, L_ECR, loss_min = losses
            
            # self.add_loss(L_ER)
            # self.add_loss(L_ECR)
            # self.add_loss(loss_min)
            self.add_loss(loss_cls)
            
            L_all = sum(losses)
            self.add_metric(L_ER, name='L_ER')
            self.add_metric(L_ECR, name='L_ECR')
            self.add_metric(loss_min, name='L_min')
            self.add_metric(loss_cls, name='L_cls')
            self.add_metric(L_all, name='L_all')
            
            pred_y_o = features[0]
            return pred_y_o
            
        else:
            return self.model(x)
            