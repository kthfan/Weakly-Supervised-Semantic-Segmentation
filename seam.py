

import numpy as np
import tensorflow as tf
from .utils.affine import *
from .utils.nonlocal_operation import *

class SEAM(tf.keras.models.Model):
    '''
    Implementation of Self-supervised Equivariant Attention Mechanism for Weakly Supervised Semantic Segmentation
    (https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Self-Supervised_Equivariant_Attention_Mechanism_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2020_paper.pdf)
    in tensorflow.
    
    # Arguments:
        image_input: The image input(s) of the model: a keras.Input object or list of keras.Input objects. 
        feature_output: The output feature of the CNN. 
        correlation_feature: Feature map for building the correlation matrix, see nonlocal neural networks.
        classify_output: Classification output of model.  # Don't use
        embedding_length: If correlation_feature is not givien, correlation_feature represents channels of correlation_feature.
        fill_mode: Fill mode of affine transformations.
        classes: Number of categories in classification task.
        ER_reg_coef: Regularization coefficient of ER loss.
        ECR_reg_coef: Regularization coefficient of ECR loss.
        epsilon: Small float added to variance to avoid dividing by zero.
        use_inverse_affine: Whether to use inverse affine transformation of feature to compute ER and ECR losses.
        min_pooling_percentage: Use top min percentage of refined cam feature map to compute min pooling loss.
                                If 0 is given, min pooling loss will be omitted.
        affine_rotate, affine_rescale, affine_flip, affine_translation: Affine transformation weights.
    
    # Usage:
        img_input = tf.keras.layers.Input((28, 28, 1))
        x = tf.keras.layers.Conv2D(64, 3, padding="same")(img_input)
        x = tf.keras.layers.BatchNormalization()(x)
        x0 = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(128, 3, padding="same")(x0)
        corr = tf.keras.layers.Concatenate()([img_input, x0])
        
        seam = SEAM(img_input, x, corr, classes=10, min_pooling_percentage=0.7)
        seam.compile(tf.keras.optimizers.Adam(learning_rate=1e-4), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
        seam.fit(X_train, y_train)
        
        pred_y = seam.predict(X_test)
        
        # save model
        seam.save("./model.h5")
        seam1 = SEAM.load("./model.h5", classes=10, min_pooling_percentage=0.7)
        
    '''
    def __init__(self, image_input, feature_output, correlation_feature=None, classify_output=None, embedding_length=64, 
                  name="seam", fill_mode='constant', classes=None, ER_reg_coef=1, ECR_reg_coef=1, epsilon=1e-6,
                 use_inverse_affine=True, min_pooling_percentage=1/4,
                affine_rotate=0, affine_rescale=0.5, affine_flip=0.5, affine_translation=0, **kwargs):
        super(SEAM, self).__init__(name=name, **kwargs)
        
        self.affine_weights = np.array([affine_rotate, 0, affine_rescale, affine_flip, affine_translation], dtype=np.float32)
        self.affine_weights /= self.affine_weights.sum()
        self.affiner = Affine(affine_weights=self.affine_weights, fill_mode=fill_mode)
        self.ER_reg_coef = ER_reg_coef
        self.ECR_reg_coef = ECR_reg_coef
        self.use_inverse_affine = use_inverse_affine
        self.min_pooling_percentage = min_pooling_percentage
        self.epsilon = epsilon
        
        if correlation_feature is None:
            correlation_feature = tf.keras.layers.Conv2D(embedding_length, 1, use_bias=False)(feature_output)
        
        if classify_output is not None and classes is None:
            classes = classify_output.shape[-1]
        
        categorical_feature, cam_feature, refined_cam_feature, classify_output, classify_output_bg = self._cnn_head(feature_output, correlation_feature, classes)
        if classify_output is None:
            classify_output = classify_output_0
        
        
        
        self.model = tf.keras.models.Model(inputs=image_input, 
                                           outputs=[feature_output, categorical_feature, 
                                                    cam_feature, refined_cam_feature, 
                                                    classify_output, classify_output_bg])
        
        # refined cam to prob saliency map
        refined_cam_feature = tf.nn.relu(refined_cam_feature) / (tf.reduce_max(refined_cam_feature, axis=(1,2), keepdims=True) + self.epsilon) # normalize
        refined_cam_feature = tf.cast(refined_cam_feature >= tf.reduce_max(refined_cam_feature, axis=-1, keepdims=True), tf.float32) * refined_cam_feature
        
        cam_feature = tf.image.resize(cam_feature, image_input.shape[1:3])
        refined_cam_feature = tf.image.resize(refined_cam_feature, image_input.shape[1:3])
        
        # models for application
        self.cam = tf.keras.models.Model(inputs=image_input, outputs=cam_feature)
        self.refined_cam = tf.keras.models.Model(inputs=image_input, outputs=refined_cam_feature)
        self.classifier = tf.keras.models.Model(inputs=image_input, outputs=classify_output)
        
        
    def _cnn_head(self, feature_output, correlation_feature, classes):

        # pointwise conv equivalents to fc layer, include last channel, which is background
        categorical_feature = tf.keras.layers.Conv2D(classes+1, 1, use_bias=False)(feature_output)
        
        # compute cam
        cam_prob = self._max_norm(categorical_feature, False) # normalize
        bg_map = 1 - tf.reduce_max(cam_prob[:, :, :, :-1], axis=-1, keepdims=True) # 1 - max(feature map among all classes)
        cam_feature = self._non_max_supress(cam_prob[:,:,:,:-1])
        cam_feature = tf.keras.layers.Concatenate()([cam_feature, bg_map])
        cam_feature = tf.stop_gradient(cam_feature)
        
        # RCM
        # ensure height and width of cam_feature and correlation_feature are equal
        r_h, r_w = max(cam_feature.shape[1], correlation_feature.shape[1]), max(cam_feature.shape[2], correlation_feature.shape[2])
        if r_h != cam_feature.shape[1] or r_w != cam_feature.shape[2]:
            cam_feature = tf.image.resize(cam_feature, (r_h, r_w))
        if r_h != correlation_feature.shape[1] or r_w != correlation_feature.shape[2]:
            correlation_feature = tf.image.resize(correlation_feature, (r_h, r_w))
        if r_h != categorical_feature.shape[1] or r_w != categorical_feature.shape[2]:
            categorical_feature = tf.image.resize(categorical_feature, (r_h, r_w))
            
        refined_cam_feature = nonlocal_operation(cam_feature, use_gaussian=False, use_relu=True, 
                               embed1=correlation_feature, embed2=correlation_feature, name="rcm")
        
        # classification
        classify_logit = tf.keras.layers.AveragePooling2D(categorical_feature.shape[1:3])(categorical_feature)
        classify_logit = tf.keras.layers.Flatten()(classify_logit)
        classify_output = tf.keras.layers.Activation("softmax" if classes>1 else 'sigmoid')(classify_logit[:, :-1])
        classify_output_bg = tf.keras.layers.Activation("softmax")(classify_logit) # include background
        
        return categorical_feature, cam_feature, refined_cam_feature, classify_output, classify_output_bg
    
    def _max_norm(self, x, include_min=False):
        if include_min:
            x = tf.nn.relu(x)
            m = tf.reduce_min(x, axis=(1,2), keepdims=True)
            return (x - m) / (tf.reduce_max(x, axis=(1,2), keepdims=True) - m + self.epsilon)
        return tf.nn.relu(x) / (tf.reduce_max(x, axis=(1,2), keepdims=True) + self.epsilon)
    
    def _non_max_supress(self, x):   
        return tf.cast(x >= tf.reduce_max(x, axis=-1, keepdims=True), x.dtype) * x
    
    def _min_pooling_loss(self, x, percentage=1/4):
        # This loss does not affect the highest performance, but change the optimial background score (alpha)
        k = int(x.shape[1]*x.shape[2]*percentage)
        x = tf.reduce_max(x, axis=-1)
        x = tf.keras.layers.Flatten()(x)
        y = -tf.nn.top_k(-x, k=k, sorted=False)[0]
        y = tf.nn.relu(y)
        return tf.reduce_sum(y, axis=-1) / k
    
    def call(self, x):
        return self.model(x)
    
    def train_step(self, data):
        x, y = data
        affine_code, affine_args, inv_affine_args = self._ranomd_affine_code(x.shape[0])
        A_x = self._affine(x, affine_code, affine_args)
        
        with tf.GradientTape() as tape:
            _, F_o, _, F_o_r, pred_y_o, _ = self(x, training=True)
            _, F_t, _, F_t_r, pred_y_t, _ = self(A_x, training=True)
            
            # only objects exist are important. (note that background always appear)
            y_bg = tf.pad(y, [[0,0],[0,1]], mode='CONSTANT', constant_values=1)
            y_bg = tf.keras.layers.Reshape((1, 1, -1))(y_bg)
            F_o, F_t, F_o_r, F_t_r = F_o*y_bg, F_t*y_bg, F_o_r*y_bg, F_t_r*y_bg
            
            # min pooling loss
            loss_min = 0
            if self.min_pooling_percentage!=0:
                loss_min_o = self._min_pooling_loss(F_o_r, self.min_pooling_percentage)
                loss_min_t = self._min_pooling_loss(F_t_r, self.min_pooling_percentage)
                loss_min = 0.5*(loss_min_o + loss_min_t)
            
            # Note that refined cam is not normalized
            # F_o_r, F_t_r = self._max_norm(F_o_r, False), self._max_norm(F_t_r, False)
            
            # spatial equivalent regularization
            A_F_o = self._affine(F_o, affine_code, affine_args)
            A_F_o_r = self._affine(F_o_r, affine_code, affine_args)
            
            L_ER = tf.abs(A_F_o - F_t)
            L_ECR = tf.abs(A_F_o_r - F_t) + tf.abs(A_F_o - F_t_r)
            
            if self.use_inverse_affine:
                inv_A_F_t = self._affine(F_t, affine_code, inv_affine_args)
                inv_A_F_t_r = self._affine(F_t_r, affine_code, inv_affine_args)
                
                L_ER += tf.abs(F_o - inv_A_F_t)
                L_ECR += tf.abs(F_o_r - inv_A_F_t) + tf.abs(F_o - inv_A_F_t_r)
                                                            
                L_ER, L_ECR = 0.5*L_ER, 0.5*L_ECR
            
            L_ER, L_ECR = tf.reduce_mean(L_ER, axis=(1,2,3)), tf.reduce_mean(L_ECR, axis=(1,2,3))
            L_ER, L_ECR = self.ER_reg_coef*L_ER, self.ECR_reg_coef*L_ECR
            L_ER, L_ECR = tf.expand_dims(L_ER, -1), tf.expand_dims(L_ECR, -1)
            
            # classification loss
            loss_cls_o = self.compiled_loss(y, pred_y_o, regularization_losses=None)
            loss_cls_t = self.compiled_loss(y, pred_y_t, regularization_losses=None)
            loss_cls = 0.5*(loss_cls_o + loss_cls_t)
            
            # other losses
            loss_reg = sum(self.losses)
            
            loss = loss_cls + L_ER + L_ECR + loss_min + loss_reg
        
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, pred_y_o)
        
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics
    
    def predict_step(self, data):
        return self.model.predict_step(data)

    def save(self, filepath, **kwds):
        return self.model.save(filepath, **kwds)
    
    @staticmethod
    def load(filepath, **kwds):
        model = tf.keras.models.load_model(filepath)
        seam = SEAM(model.inputs[0], model.outputs[0], model.outputs[2], **kwds)
        seam.model = model
        return seam
    
    def _ranomd_affine_code(self, size):
        if size is None:
            return None, None, None
        code, args = self.affiner.random_affine_code(size)
        inv_args = self.affiner.inverse_affine_code(code, args)
        
        return code, args, inv_args
    
    def _affine(self, I, code, args):
        if code is None:
            return I
        A_I = self.affiner.apply_affine(I, code, args)
        A_I = tf.stop_gradient(A_I)
        return A_I
    