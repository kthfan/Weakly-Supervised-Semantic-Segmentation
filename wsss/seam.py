

import numpy as np
import tensorflow as tf
from .utils import Affine
from .utils import nonlocal_neural_network

class SEAM(tf.keras.models.Model):
    '''
    Implementation of Self-supervised Equivariant Attention Mechanism for Weakly Supervised Semantic Segmentation
    (https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Self-Supervised_Equivariant_Attention_Mechanism_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2020_paper.pdf)
    in tensorflow.
    
    # Arguments:
        image_input: The image input(s) of the model: a keras.Input object or list of keras.Input objects. 
        feature_output: The output feature of the CNN. 
        classes: Number of categories in classification task.
        correlation_feature: Feature map for building the correlation matrix, see nonlocal neural networks. 
                                Note that no gradient will be provided to the top weights.
        classification_activation: Activation function on classification output. Default is set to 'sigmoid'.
        fill_mode: Fill mode of affine transformations.
        ER_reg_coef: Regularization coefficient of ER loss.
        ECR_reg_coef: Regularization coefficient of ECR loss.
        epsilon: Small float added to variance to avoid dividing by zero.
        use_inverse_affine: Whether to use inverse affine transformation of feature to compute ER and ECR losses.
        min_pooling_rate: Use top min percentage of refined cam feature map to compute min pooling loss.
                                If 0 is given, min pooling loss will be omitted.
        ECR_pooling_rate: Define how many pixels will be used to compute ECR loss.
                            The higher pixels with higher loss will be selected first.
        affine_rotate, affine_rescale, affine_flip, affine_translation: Affine transformation weights.
    
    # Usage:
        img_input = tf.keras.layers.Input((28, 28, 1))
        x = tf.keras.layers.Conv2D(64, 3, padding="same")(img_input)
        x = tf.keras.layers.BatchNormalization()(x)
        x0 = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(128, 3, padding="same")(x0)
        corr = tf.keras.layers.Concatenate()([img_input, x0])
        
        seam = SEAM(img_input, x, 10, correlation_feature=corr, min_pooling_rate=0.7, classification_activation="softmax")
        seam.compile(tf.keras.optimizers.Adam(learning_rate=1e-4), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
        seam.fit(X_train, y_train)
        
        pred_y = seam.predict(X_test)
        
        # save and load model
        config = seam.get_config()
        seam.save("./model.h5")
        seam1 = SEAM.load_model("./model.h5", **config)
        
    '''
    def __init__(self, image_input, feature_output, classes, correlation_feature=None,
                  classification_activation="sigmoid", fill_mode='constant', ER_reg_coef=1, ECR_reg_coef=1, epsilon=1e-6,
                 use_inverse_affine=True, min_pooling_rate=1/4, ECR_pooling_rate=0.2, name="seam", 
                affine_rotate=0, affine_rescale=0.5, affine_flip=0.5, affine_translation=0, **kwargs):
        super(SEAM, self).__init__(name=name, **kwargs)
        
        self.affine_weights = np.array([affine_rotate, 0, affine_rescale, affine_flip, affine_translation], dtype=np.float32)
        self.affine_weights /= self.affine_weights.sum()
        self.affiner = Affine(affine_weights=self.affine_weights, fill_mode=fill_mode, 
                              vertical_flip=False, rotate_domain=(-np.pi/9, np.pi/9), rescale_scale_domain=(0.7, 1.3),
                             translation_y_domain=(-0.033, 0.033), translation_x_domain=(-0.033, 0.033))
        self.ER_reg_coef = ER_reg_coef
        self.ECR_reg_coef = ECR_reg_coef
        self.use_inverse_affine = use_inverse_affine
        self.min_pooling_rate = min_pooling_rate
        self.ECR_pooling_rate = ECR_pooling_rate
        self.classification_activation = classification_activation
        self.epsilon = epsilon
        
        if type(self) == SEAM:
            outputs = self._cnn_head(feature_output, correlation_feature, classes)
            self._build_models(image_input, feature_output, *outputs)
        
    def _cnn_head(self, feature_output, correlation_feature, classes):
        if correlation_feature is None:
            correlation_feature = tf.keras.layers.Conv2D(192, 1, use_bias=False)(tf.stop_gradient(feature_output))
        
        # pointwise conv equivalents to fc layer, include last channel, which is background
        categorical_feature = tf.keras.layers.Conv2D(classes, 1, use_bias=False)(feature_output)
        
        # classification
        classify_logit = tf.keras.layers.AveragePooling2D(categorical_feature.shape[1:3])(categorical_feature)
        classify_logit = tf.keras.layers.Flatten()(classify_logit)
        classify_output = tf.keras.layers.Activation('sigmoid 'if self.classification_activation=="softmax" and classes==1 else self.classification_activation)(classify_logit)
        
        # compute cam
        cam_prob = self._max_norm(categorical_feature, False) # normalize
        bg_map = 1 - tf.reduce_max(cam_prob, axis=-1, keepdims=True) # 1 - max(feature map among all classes)
        cam_feature = self._non_max_suppress(cam_prob)
        cam_feature = tf.keras.layers.Concatenate()([cam_feature, bg_map])
        cam_feature = tf.stop_gradient(cam_feature)
        
        # RCM
        # ensure height and width of cam_feature and correlation_feature are equal
        r_h, r_w = correlation_feature.shape[1], correlation_feature.shape[2]
        if r_h != cam_feature.shape[1] or r_w != cam_feature.shape[2]:
            cam_feature = tf.image.resize(cam_feature, (r_h, r_w))
            categorical_feature = tf.image.resize(categorical_feature, (r_h, r_w))   
            
        refined_cam_feature = nonlocal_neural_network(cam_feature, use_gaussian=False, use_relu=True, 
                               embed1=correlation_feature, embed2=correlation_feature, name="rcm")
        
        
        return categorical_feature, cam_feature, refined_cam_feature, classify_output
    
    def _build_models(self, image_input, feature_output, categorical_feature, 
                      cam_feature, refined_cam_feature, classify_output):
        
        self.model = tf.keras.models.Model(inputs=image_input, 
                                           outputs=[feature_output, categorical_feature, 
                                                    cam_feature, refined_cam_feature, 
                                                    classify_output])
        
        # refined cam to prob saliency map
        refined_cam_feature = tf.nn.relu(refined_cam_feature) / (tf.reduce_max(refined_cam_feature, axis=(1,2), keepdims=True) + self.epsilon) # normalize
        refined_cam_feature = self._non_max_suppress(refined_cam_feature)
        
        cam_feature = tf.image.resize(cam_feature, image_input.shape[1:3])
        refined_cam_feature = tf.image.resize(refined_cam_feature, image_input.shape[1:3])
        
        # models for application
        self.cam = tf.keras.models.Model(inputs=image_input, outputs=cam_feature)
        self.refined_cam = tf.keras.models.Model(inputs=image_input, outputs=refined_cam_feature)
        self.classifier = tf.keras.models.Model(inputs=image_input, outputs=classify_output)
    
    def _max_norm(self, x, include_min=False):
        if include_min:
            x = tf.nn.relu(x)
            m = tf.reduce_min(x, axis=(1,2), keepdims=True)
            return (x - m) / (tf.reduce_max(x, axis=(1,2), keepdims=True) - m + self.epsilon)
        return tf.nn.relu(x) / (tf.reduce_max(x, axis=(1,2), keepdims=True) + self.epsilon)
    
    def _non_max_suppress(self, x, g=None):  
        if g is None:
            g = x
        mask = tf.cast(x >= tf.reduce_max(x, axis=-1, keepdims=True), x.dtype)
        return mask * g
    
    def _min_pooling_loss(self, x, rate=None):
        # This loss does not affect the highest performance, but change the optimial background score (alpha)
        if rate is None:
            rate = self.min_pooling_rate
            
        k = int(x.shape[1]*x.shape[2]*rate)
        x = tf.reduce_max(x, axis=-1)
        x = tf.keras.layers.Flatten()(x)
        y = -tf.nn.top_k(-x, k=k, sorted=False)[0]
        y = tf.nn.relu(y)
        return tf.reduce_sum(y, axis=-1) / k
    
    def _loss(self, affine_code, affine_args, inv_affine_args, x, y, A_x, 
              F_o, F_t, cam_o, cam_t, F_o_r, F_t_r, pred_y_o, pred_y_t):
            
        # only objects exist are important. (note that background always appear)
        y_bg = tf.pad(y, [[0,0],[0,1]], mode='CONSTANT', constant_values=1)
        y_bg = tf.keras.layers.Reshape((1, 1, -1))(y_bg)
        _y = tf.keras.layers.Reshape((1, 1, -1))(y)
        cam_o, cam_t, F_o_r, F_t_r = cam_o*y_bg, cam_t*y_bg, F_o_r*y_bg, F_t_r*y_bg
        F_o, F_t = F_o*_y, F_t*_y
            
        # min pooling loss
        loss_min = 0
        if self.min_pooling_rate!=0:
            loss_min_o = self._min_pooling_loss(F_o_r[:, :, :, :-1])
            loss_min_t = self._min_pooling_loss(F_t_r[:, :, :, :-1])
            loss_min_o, loss_min_t = tf.reduce_mean(loss_min_o), tf.reduce_mean(loss_min_t)
            loss_min = 0.5*(loss_min_o + loss_min_t)
                
        # spatial equivalent regularization
        A_F_o = self.affiner.apply_affine(F_o, affine_code, affine_args)
        A_cam_o = self.affiner.apply_affine(cam_o, affine_code, affine_args)
        A_F_o_r = self.affiner.apply_affine(F_o_r, affine_code, affine_args)
            
        L_ER = tf.abs(A_F_o - F_t)
        L_ECR = tf.abs(A_F_o_r - self._non_max_suppress(cam_t)) + tf.abs(self._non_max_suppress(A_cam_o) - F_t_r)
            
        inv_A_F_t, inv_A_F_t_r = None, None
        if self.use_inverse_affine:
            inv_A_F_t = self.affiner.apply_affine(F_t, affine_code, inv_affine_args)
            inv_A_cam_t = self.affiner.apply_affine(cam_t, affine_code, inv_affine_args)
            inv_A_F_t_r = self.affiner.apply_affine(F_t_r, affine_code, inv_affine_args)
                
            L_ER += tf.abs(F_o - inv_A_F_t)
            L_ECR += tf.abs(F_o_r - self._non_max_suppress(inv_A_cam_t)) + tf.abs(self._non_max_suppress(cam_o) - inv_A_F_t_r)
                                                            
            L_ER, L_ECR = 0.5*L_ER, 0.5*L_ECR
          
        # only pooling the L_ECR which is significant.
        L_ECR = tf.keras.layers.Flatten()(L_ECR)
        if self.ECR_pooling_rate != 1:
            L_ECR = tf.nn.top_k(L_ECR, k=int(L_ECR.shape[1]*self.ECR_pooling_rate), sorted=False)[0]
            
        L_ER, L_ECR = tf.reduce_mean(L_ER, axis=(0,1,2,3)), tf.reduce_mean(L_ECR, axis=(0,1))
        L_ER, L_ECR = self.ER_reg_coef*L_ER, self.ECR_reg_coef*L_ECR
            
        # classification loss
        loss_cls_o = self.compiled_loss(y, pred_y_o, regularization_losses=None)
        loss_cls_t = self.compiled_loss(y, pred_y_t, regularization_losses=None)
        loss_cls = 0.5*(loss_cls_o + loss_cls_t)
           
        return ((pred_y_o, pred_y_t, y_bg, F_o, F_t, F_o_r, F_t_r, A_F_o, A_F_o_r, inv_A_F_t, inv_A_F_t_r),
                (loss_cls, L_ER, L_ECR, loss_min))
            
    def call(self, x, y=None, training=False):
        if training and y is not None:
            # apply affine
            affine_code, affine_args, inv_affine_args = self._ranomd_affine_code(tf.shape(x)[0])
            A_x = self.affiner.apply_affine(x, affine_code, affine_args)
            
            # get features from simsiam model
            _, F_o, cam_o, F_o_r, pred_y_o = self.model(x, training=True)
            _, F_t, cam_t, F_t_r, pred_y_t = self.model(A_x, training=True)
        
            # compute losses
            features, losses = self._loss(affine_code, affine_args, inv_affine_args, x, y, A_x, F_o, F_t, cam_o, cam_t, F_o_r, F_t_r, pred_y_o, pred_y_t)
            
            loss_cls, L_ER, L_ECR, loss_min = losses
            
            self.add_loss(L_ER)
            self.add_loss(L_ECR)
            self.add_loss(loss_min)
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
    
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            pred_y_o = self(x, y=y, training=True)
            
            loss = sum(self.losses)
        
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

    def compile(self, optimizer, **kwargs):
#         self.refined_cam.compile(optimizer, loss=, metrics=kwargs.get('metrics', ['miou']))
#         self.cam.compile(optimizer, loss=, metrics=kwargs.get('metrics', ['miou']))
        self.classifier.compile(optimizer, **kwargs)
        super(SEAM, self).compile(optimizer=optimizer, **kwargs)
        
    def get_config(self):
        
        return {"name": self.name,
                "classes": self.model.outputs[4].shape[-1],
                "classification_activation": self.classification_activation,
                "fill_mode": self.affiner.fill_mode,
                "ER_reg_coef": self.ER_reg_coef,
                "ECR_reg_coef": self.ECR_reg_coef,
                "epsilon": self.epsilon,
                "use_inverse_affine": self.use_inverse_affine,
                "min_pooling_rate": self.min_pooling_rate,
                "affine_rotate": self.affine_weights[0],
                "affine_rescale": self.affine_weights[2],
                "affine_flip": self.affine_weights[3],
                "affine_translation": self.affine_weights[4],
                "model": self.model.get_config()
               }
    
    def save(self, filepath, **kwds):
        return self.model.save(filepath, **kwds)
    
    @staticmethod
    def from_config(config):
        model = tf.keras.models.Model.from_config(config.pop("model"))
        image_input = model.inputs[0]
        feature_output, categorical_feature, cam_feature, refined_cam_feature, classify_output = model.outputs
        seam = SEAM(image_input, feature_output, **config)
        seam._build_models(image_input, feature_output, categorical_feature, 
                      cam_feature, refined_cam_feature, classify_output)
        return seam
    
    @staticmethod
    def load_model(filepath, **kwds):
        model = tf.keras.models.load_model(filepath)
        image_input = model.inputs[0]
        feature_output, categorical_feature, cam_feature, refined_cam_feature, classify_output = model.outputs
        seam = SEAM(image_input, feature_output, **kwds)
        seam._build_models(image_input, feature_output, categorical_feature, 
                      cam_feature, refined_cam_feature, classify_output)
        return seam
    
    def _ranomd_affine_code(self, size):
        code, args = self.affiner.random_affine_code(size)
        inv_args = self.affiner.inverse_affine_code(code, args)
        
        return code, args, inv_args
    