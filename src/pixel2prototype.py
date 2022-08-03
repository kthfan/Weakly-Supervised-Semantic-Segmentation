import numpy as np
import tensorflow as tf
from .seam import *


class Pixel2Prototype(SEAM):
    '''
    Implementation of Weakly Supervised Semantic Segmentation by Pixel-to-Prototype Contrast
    (https://openaccess.thecvf.com/content/CVPR2022/papers/Du_Weakly_Supervised_Semantic_Segmentation_by_Pixel-to-Prototype_Contrast_CVPR_2022_paper.pdf)
    in tensorflow.
    
    # Arguments:
        image_input: The image input(s) of the model: a keras.Input object or list of keras.Input objects. 
        feature_output: The output feature of the CNN. 
        classes: Number of categories in classification task.
        project_feature: The pixel-wise projected feature for prototype contrastive learning. 
        prototype_confidence_rate: Use top K confidences project_feature to estimate the prototype.
        nce_tau: The temperature coefficient in nce loss function.
        cp_contrast_coef: Coefficient of cross prototype contrast loss.
        cc_contrast_coef: Coefficient of cross CAM contrast loss.
        intra_contrast_coef: Coefficient of intra-view contrast loss.
        hard_prototype_range: Upper bound and lower bound of hard prototype percentage.
        hard_pixel_range:  Upper bound and lower bound of hard pixel percentage.
        
    # Usage:
        img_input = tf.keras.layers.Input((28, 28, 1))
        x = tf.keras.layers.Conv2D(64, 3, padding="same")(img_input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(128, 3, padding="same")(x)
        proj = tf.keras.layers.Conv2D(128, 1)(x)
        
        p2p = Pixel2Prototype(img_input, x, 10, project_feature=proj)
        p2p.compile(tf.keras.optimizers.Adam(learning_rate=1e-4), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
        p2p.fit(X_train, y_train)
    '''
    def __init__(self, image_input, feature_output, classes, project_feature=None, correlation_feature=None,  
                 name="p2p", prototype_confidence_rate=1/8, nce_tau=0.1, cp_contrast_coef=0.1, cc_contrast_coef=0.1,
                 intra_contrast_coef=0.1, hard_prototype_range=(0.1, 0.6), hard_pixel_range=(0.1, 0.6), **kwargs):
        super(Pixel2Prototype, self).__init__(image_input, feature_output, classes, name=name, **kwargs)
                        
        self.prototype_confidence_rate = prototype_confidence_rate
        self.nce_tau = nce_tau
        self.cp_contrast_coef = cp_contrast_coef
        self.cc_contrast_coef = cc_contrast_coef
        self.intra_contrast_coef = intra_contrast_coef
        self.hard_prototype_range = hard_prototype_range
        self.hard_pixel_range = hard_pixel_range
        
        if type(self) == Pixel2Prototype:
            outputs = self._cnn_head(feature_output, correlation_feature, project_feature, classes)
            self._build_models(image_input, feature_output, *outputs)   
        
    def _cnn_head(self, feature_output, correlation_feature, project_feature, classes):
        if project_feature is None:
            project_feature = tf.keras.layers.Conv2D(128, 1, use_bias=False)(feature_output)
            
        categorical_feature, cam_feature, refined_cam_feature, classify_output, classify_output_bg = super(Pixel2Prototype, self)._cnn_head(feature_output, correlation_feature, classes)
        return categorical_feature, cam_feature, refined_cam_feature, project_feature, classify_output, classify_output_bg
    
    def _build_models(self, image_input, feature_output, categorical_feature, 
                      cam_feature, refined_cam_feature, project_feature, 
                      classify_output, classify_output_bg):
        
        super(Pixel2Prototype, self)._build_models(image_input, feature_output, categorical_feature, cam_feature, refined_cam_feature, classify_output, classify_output_bg )
        
        self.model = tf.keras.models.Model(inputs=image_input, 
                                           outputs=[feature_output, categorical_feature, 
                                                    cam_feature, refined_cam_feature, project_feature, 
                                                    classify_output, classify_output_bg])
        
    def _get_prototype(self, cam_feature, project_feature, rate=None):
        if rate is None:
            rate = self.prototype_confidence_rate
        
        batch_size, h, w, n_class = cam_feature.shape
        n_channel = project_feature.shape[-1]
        k = int(h*w*rate)
        
        assert h == project_feature.shape[1] and w == project_feature.shape[2]
        
        # reshape to top_k()
        cam_feature = tf.transpose(cam_feature, (3, 0, 1, 2)) # (n_class, batch_size, h, w)
        cam_feature = tf.reshape(cam_feature, (n_class, -1)) # (n_class, batch_size*h*w)

        # compute top k index of cam_feature
        categories_score, categories_index = tf.nn.top_k(cam_feature, k=k, sorted=False) # (n_class, k)
        categories_index = tf.stop_gradient(categories_index)
        
        # get interested project_feature by top k index of cam feature
        project_feature = tf.reshape(project_feature, (-1, n_channel)) # (batch_size*h*w, n_channel)
        prototype_feature = tf.gather(project_feature, categories_index, axis=0) # (n_class, k, n_channel)
        
        # compute prototype
        categories_score = tf.expand_dims(categories_score, -1) # (n_class, k, 1)
        prototype = tf.reduce_sum(prototype_feature*categories_score, axis=1) # (n_class, n_channel)
        prototype = prototype / (tf.reduce_sum(categories_score, axis=1) + self.epsilon)
        
        return prototype
    
   
    def _nce_loss(self, feature, label, prototype, tau=None):
        if tau is None:
            tau = self.nce_tau
            
        positive = tf.gather(prototype, label, axis=0) # (batch_size, n_channel)
        positive = tf.reduce_sum(feature*positive, axis=-1) / tau # (batch_size)
        negative = feature @ tf.transpose(prototype, (1, 0)) / tau # (batch_size, n_class)
        loss = tf.exp(positive) / (tf.reduce_sum(tf.exp(negative), axis=-1) + self.epsilon)
        loss = -tf.math.log(loss + self.epsilon)
        return loss
    
    def _nce_loss_hard_mining(self, feature, label, prototype, 
                              tau=None, hard_prototype_range=None, hard_pixel_range=None):
        if tau is None:
            tau = self.nce_tau
        if hard_prototype_range is None:
            hard_prototype_range = self.hard_prototype_range
        if hard_pixel_range is None:
            hard_pixel_range = self.hard_pixel_range
        
        batch_size, n_channel, n_class = feature.shape[0], feature.shape[1], prototype.shape[0]
        hard_prototype_range_0 = int(hard_prototype_range[0]*n_class)
        hard_prototype_range_1 = int(hard_prototype_range[1]*n_class)
        
        positive = tf.gather(prototype, label, axis=0) # (batch_size, n_channel)
        positive = tf.reduce_sum(feature * positive, axis=-1) # (batch_size,)
        negative = feature @ tf.transpose(prototype, (1, 0)) # (batch_size, n_class)
        
        # Semi-hard Prototype Mining
        semi_negative, _ = tf.nn.top_k(negative, k=hard_prototype_range_1, sorted=False) # (batch_size, hard_prototype_range[1])
        semi_negative = semi_negative[:, hard_prototype_range_0:] # (batch_size, k0)
        semi_negative = tf.concat([tf.expand_dims(positive, axis=-1), semi_negative], axis=-1) # add positive class # (# (batch_size, k0+1))
        
        semi_loss = tf.exp(positive) / (tf.reduce_sum(tf.exp(semi_negative), axis=-1) + self.epsilon)
        semi_loss = -tf.math.log(semi_loss + self.epsilon) # (batch_size,)
        
        # Hard Pixel Sampling
        similarity = positive - 1 # in [-2, 0]
        
        ## filiter pixels for each category by given label
        category_mask = tf.cast(label == tf.expand_dims(tf.range(0, n_class, dtype=label.dtype), -1), 
                                dtype=similarity.dtype) # (n_class, batch_size)
        category_pixels = tf.math.count_nonzero(category_mask, axis=-1) # (n_class,)
        
        ## get reasonable similarity for each category
        similarity = tf.transpose(category_mask * similarity, (1, 0)) # (batch_size, n_class)
        similarity = tf.boolean_mask(similarity, category_pixels>1, axis=1) # (batch_size, |n_class that has more than 1 pixel|)
        
        ## category_pixels: category_pixels[category_pixels>1]; hard_pixel_index: to get top k dissimilar pixels; hard_pixel_loss: semi_loss[hard_pixel_index]
        category_pixels = tf.boolean_mask(category_pixels, category_pixels>1, axis=0) # (|n_class that has more than 1 pixel|,)
        hard_pixel_index = tf.argsort(similarity, axis=0, direction='ASCENDING') # (batch_size, |category_pixels|)
        hard_pixel_index = tf.stop_gradient(hard_pixel_index)
        hard_pixel_loss = tf.gather(semi_loss, hard_pixel_index) # (batch_size, |category_pixels|)
        
        ## random pick |category_pixels| * (1 - hard_pixel_range[1] + hard_pixel_range[0])
        random_bound = tf.cast(tf.math.ceil(tf.cast(category_pixels, tf.float32) * (1 - hard_pixel_range[1] + hard_pixel_range[0])), category_pixels.dtype)
        random_pixel_loss = tf.map_fn(
                lambda i: tf.reduce_mean(
                    tf.gather(               # get shuffled top category_pixels[i] pixel losses
                        hard_pixel_loss[:, i], 
                        tf.random.shuffle(tf.range(category_pixels[i])) # shuffle indices of hard_pixel_loss
                    )[:random_bound[i]]      # get random_bound[i] shuffled pixel losses
                ), 
                tf.range(tf.shape(category_pixels)[0], dtype=category_pixels.dtype),
            dtype=hard_pixel_loss.dtype) # (|category_pixels|, )
        random_pixel_loss = tf.reduce_mean(random_pixel_loss)
        
        ## select hard_pixel_loss between (hard_pixel_range[0], hard_pixel_range[1])
        upper_bound = tf.cast(tf.cast(category_pixels, tf.float32) * hard_pixel_range[1], category_pixels.dtype)
        lower_bound = tf.cast(tf.cast(category_pixels, tf.float32) * hard_pixel_range[0], category_pixels.dtype)
        hard_pixel_loss = tf.map_fn(lambda i: tf.reduce_mean(hard_pixel_loss[lower_bound[i]:upper_bound[i], i]), 
                                    tf.range(tf.shape(category_pixels)[0], dtype=category_pixels.dtype),
                                   dtype=hard_pixel_loss.dtype) # (|category_pixels|, )
        hard_pixel_loss = tf.reduce_mean(hard_pixel_loss)
        
        # overall loss
        loss = (1 - hard_pixel_range[1] + hard_pixel_range[0])*random_pixel_loss + (hard_pixel_range[1] - hard_pixel_range[0])*hard_pixel_loss
        return loss
            
    def call(self, x, y=None, training=False):
        if training and y is not None:
            affine_code, affine_args, inv_affine_args = self._ranomd_affine_code(tf.shape(x)[0])
            A_x = self._affine(x, affine_code, affine_args)
            
            _, F_o, _, F_o_r, F_proj_o, pred_y_o, _ = self.model(x, training=True)
            _, F_t, _, F_t_r, F_proj_t, pred_y_t, _ = self.model(A_x, training=True)
            
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
            L_ECR = tf.abs(A_F_o_r - F_t) + tf.abs(A_F_o - F_t_r)
            
            if self.use_inverse_affine:
                inv_A_F_t = self._affine(F_t, affine_code, inv_affine_args)
                inv_A_F_t_r = self._affine(F_t_r, affine_code, inv_affine_args)
                
                L_ER += tf.abs(F_o - inv_A_F_t)
                L_ECR += tf.abs(F_o_r - inv_A_F_t) + tf.abs(F_o - inv_A_F_t_r)
                                                            
                L_ER, L_ECR = 0.5*L_ER, 0.5*L_ECR
            
            L_ER, L_ECR = tf.reduce_mean(L_ER, axis=(0,1,2,3)), tf.reduce_mean(L_ECR, axis=(0,1,2,3))
            L_ER, L_ECR = self.ER_reg_coef*L_ER, self.ECR_reg_coef*L_ECR
            
            # contrastive learning loss
            F_proj_o = tf.linalg.normalize(F_proj_o, axis=-1)[0]
            F_proj_t = tf.linalg.normalize(F_proj_t, axis=-1)[0]
            pseudo_label_o = tf.argmax(F_o_r, axis=-1) # (batch_size, h, w)
            pseudo_label_t = tf.argmax(F_t_r, axis=-1) # (batch_size, h, w)
            prototype_o = self._get_prototype(F_o_r, F_proj_o, self.prototype_confidence_rate) # (n_class, n_channel)
            prototype_t = self._get_prototype(F_t_r, F_proj_t, self.prototype_confidence_rate) # (n_class, n_channel)
            
            pseudo_label_o, pseudo_label_t = tf.stop_gradient(pseudo_label_o), tf.stop_gradient(pseudo_label_t)
            prototype_o, prototype_t = tf.stop_gradient(prototype_o), tf.stop_gradient(prototype_t)
            
            ## flatten (batch_size, h, w)
            F_proj_o = tf.reshape(F_proj_o, (-1, F_proj_o.shape[-1]))
            F_proj_t = tf.reshape(F_proj_t, (-1, F_proj_o.shape[-1]))
            pseudo_label_o = tf.reshape(pseudo_label_o, [-1])
            pseudo_label_t = tf.reshape(pseudo_label_t, [-1])
            
            L_cp = 0.5*(self._nce_loss(F_proj_o, pseudo_label_o, prototype_t) + self._nce_loss(F_proj_t, pseudo_label_t, prototype_o))
            L_cc = 0.5*(self._nce_loss(F_proj_o, pseudo_label_t, prototype_o) + self._nce_loss(F_proj_t, pseudo_label_o, prototype_t))
            L_cp, L_cc = self.cp_contrast_coef*tf.reduce_mean(L_cp), self.cc_contrast_coef*tf.reduce_mean(L_cc)
            
            
            L_intra = 0.5*(self._nce_loss_hard_mining(F_proj_o, pseudo_label_o, prototype_o) + self._nce_loss_hard_mining(F_proj_t, pseudo_label_t, prototype_t))
            L_intra = self.intra_contrast_coef*L_intra # already scalar
            
            # classification loss
            loss_cls_o = self.compiled_loss(y, pred_y_o, regularization_losses=None)
            loss_cls_t = self.compiled_loss(y, pred_y_t, regularization_losses=None)
            loss_cls = 0.5*(loss_cls_o + loss_cls_t)
            
            loss = loss_cls + L_ER + L_ECR + loss_min + L_cp + L_cc + L_intra
            
            self.add_loss(L_ER)
            self.add_loss(L_ECR)
            self.add_loss(loss_min)
            self.add_loss(L_cp)
            self.add_loss(L_cc)
            self.add_loss(L_intra)
            
            self.add_metric(L_ER, name='L_ER')
            self.add_metric(L_ECR, name='L_ECR')
            self.add_metric(loss_min, name='L_min')
            self.add_metric(loss_cls, name='L_cls')
            self.add_metric(L_cp, name='L_cp')
            self.add_metric(L_cc, name='L_cc')
            self.add_metric(L_intra, name='L_intra')
            self.add_metric(loss, name='L_all')
            
            return pred_y_o
            
        else:
            return self.model(x)
        
