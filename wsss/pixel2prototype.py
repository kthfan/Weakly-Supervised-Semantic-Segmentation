
import tensorflow as tf
from .seam import SEAM


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
        background_threshold: In contrastive learning, pixel that max(cam) < background_threshold will be considered as background pixel.
                              If None, persist original background score.
    # Usage:
        img_input = tf.keras.layers.Input((224, 224, 3))
        x = tf.keras.layers.Conv2D(64, 3, padding="same")(img_input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(128, 3, padding="same")(x)
        proj = tf.keras.layers.Conv2D(128, 1)(x)
        
        p2p = Pixel2Prototype(img_input, x, 10, project_feature=proj)
        p2p.compile(tf.keras.optimizers.Adam(learning_rate=1e-4), loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])
        p2p.fit(X_train, y_train)
    '''
    def __init__(self, image_input, feature_output, classes, project_feature=None, correlation_feature=None,  
                 name="p2p", prototype_confidence_rate=1/8, nce_tau=0.1, cp_contrast_coef=0.1, cc_contrast_coef=0.1,
                 intra_contrast_coef=0.1, background_threshold=0.2,
                 hard_prototype_range=(0.1, 0.6), hard_pixel_range=(0.1, 0.6), **kwargs):
        super(Pixel2Prototype, self).__init__(image_input, feature_output, classes, name=name, **kwargs)
                        
        self.prototype_confidence_rate = prototype_confidence_rate
        self.nce_tau = nce_tau
        self.cp_contrast_coef = cp_contrast_coef
        self.cc_contrast_coef = cc_contrast_coef
        self.intra_contrast_coef = intra_contrast_coef
        self.hard_prototype_range = hard_prototype_range
        self.hard_pixel_range = hard_pixel_range
        self.background_threshold = background_threshold
        
        if type(self) == Pixel2Prototype:
            outputs = self._cnn_head(feature_output, correlation_feature, project_feature, classes)
            self._build_models(image_input, feature_output, *outputs)   
        
    def _cnn_head(self, feature_output, correlation_feature, project_feature, classes):
        if project_feature is None:
            project_feature = tf.keras.layers.Conv2D(128, 1, use_bias=False)(feature_output)
            
        categorical_feature, cam_feature, refined_cam_feature, classify_output = super(Pixel2Prototype, self)._cnn_head(feature_output, correlation_feature, classes)
           
        return categorical_feature, cam_feature, refined_cam_feature, project_feature, classify_output
    
    def _build_models(self, image_input, feature_output, categorical_feature, 
                      cam_feature, refined_cam_feature, project_feature, 
                      classify_output):
        
        super(Pixel2Prototype, self)._build_models(image_input, feature_output, categorical_feature, cam_feature, refined_cam_feature, classify_output )
        
        self.model = tf.keras.models.Model(inputs=image_input, 
                                           outputs=[feature_output, categorical_feature, 
                                                    cam_feature, refined_cam_feature, project_feature, 
                                                    classify_output])
        
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
            fn_output_signature=hard_pixel_loss.dtype) # (|category_pixels|, )
        random_pixel_loss = tf.reduce_mean(random_pixel_loss)
        
        ## select hard_pixel_loss between (hard_pixel_range[0], hard_pixel_range[1])
        upper_bound = tf.cast(tf.cast(category_pixels, tf.float32) * hard_pixel_range[1], category_pixels.dtype)
        lower_bound = tf.cast(tf.cast(category_pixels, tf.float32) * hard_pixel_range[0], category_pixels.dtype)
        hard_pixel_loss = tf.map_fn(lambda i: tf.reduce_mean(hard_pixel_loss[lower_bound[i]:upper_bound[i], i]), 
                                    tf.range(tf.shape(category_pixels)[0], dtype=category_pixels.dtype),
                                   fn_output_signature=hard_pixel_loss.dtype) # (|category_pixels|, )
        hard_pixel_loss = tf.reduce_mean(hard_pixel_loss)
        
        # overall loss
        loss = (1 - hard_pixel_range[1] + hard_pixel_range[0])*random_pixel_loss + (hard_pixel_range[1] - hard_pixel_range[0])*hard_pixel_loss
        return loss
    
    def _loss(self, affine_code, affine_args, inv_affine_args, x, y, A_x, 
              F_o, F_t, cam_o, cam_t, F_o_r, F_t_r, F_proj_o, F_proj_t, pred_y_o, pred_y_t):
        
        # features and losses from SEAM
        features, losses = super(Pixel2Prototype, self)._loss(affine_code, affine_args, inv_affine_args, x, y, A_x, F_o, F_t, cam_o, cam_t, F_o_r, F_t_r, pred_y_o, pred_y_t)
            
        pred_y_o, _, _, _, _, F_o_r, F_t_r, A_F_o, A_F_o_r, inv_A_F_t, inv_A_F_t_r = features
        loss_cls, L_ER, L_ECR, loss_min = losses      
        
        # ensure height and width of F_o_r and F_proj_o are equal
        if F_proj_o.shape[1] != F_o_r.shape[1] or F_proj_o.shape[2] != F_o_r.shape[2]:
            F_o_r = tf.image.resize(F_o_r, (F_proj_o.shape[1], F_proj_o.shape[2]))
            F_t_r = tf.image.resize(F_t_r, (F_proj_t.shape[1], F_proj_t.shape[2]))
            A_F_o_r = tf.image.resize(A_F_o_r, (F_proj_o.shape[1], F_proj_o.shape[2]))
            if self.use_inverse_affine:
                inv_A_F_t_r = tf.image.resize(inv_A_F_t_r, (F_proj_t.shape[1], F_proj_t.shape[2]))
        
        # contrastive learning loss
        ## set background score
        cam_o, cam_t = F_o_r, F_t_r
        A_cam_o, inv_A_cam_t = A_F_o_r, inv_A_F_t_r
        if self.background_threshold is not None:
            cam_o = tf.pad(F_o_r[:, :, :, :-1], [[0,0], [0,0], [0,0], [0, 1]], mode="CONSTANT", constant_values=self.background_threshold)
            cam_t = tf.pad(F_t_r[:, :, :, :-1], [[0,0], [0,0], [0,0], [0, 1]], mode="CONSTANT", constant_values=self.background_threshold)
            A_cam_o = tf.pad(A_F_o_r[:, :, :, :-1], [[0,0], [0,0], [0,0], [0, 1]], mode="CONSTANT", constant_values=self.background_threshold)
            if self.use_inverse_affine:
                inv_A_cam_t = tf.pad(inv_A_F_t_r[:, :, :, :-1], [[0,0], [0,0], [0,0], [0, 1]], mode="CONSTANT", constant_values=self.background_threshold)
        
        ## normalized projection feature
        F_proj_o = tf.linalg.normalize(F_proj_o, axis=-1)[0]
        F_proj_t = tf.linalg.normalize(F_proj_t, axis=-1)[0]
        ### Apply affine in project_feature F_proj_o and F_proj_t
        A_F_proj_o = self.affiner.apply_affine(F_proj_o, affine_code, affine_args)
        inv_A_F_proj_t = self.affiner.apply_affine(F_proj_t, affine_code, inv_affine_args) if self.use_inverse_affine else None
        
        ## pseudo pixel-level label
        pseudo_label_o = tf.argmax(cam_o, axis=-1) # (batch_size, h, w)
        pseudo_label_t = tf.argmax(cam_t, axis=-1) # (batch_size, h, w)
        A_pseudo_label_o = tf.argmax(A_cam_o, axis=-1) # (batch_size, h, w)
        inv_A_pseudo_label_t = tf.argmax(inv_A_cam_t, axis=-1) if self.use_inverse_affine else None
        
        ## prototype
        prototype_o = self._get_prototype(cam_o, F_proj_o, self.prototype_confidence_rate) # (n_class, n_channel)
        prototype_t = self._get_prototype(cam_t, F_proj_t, self.prototype_confidence_rate) # (n_class, n_channel)
        A_prototype_o = self._get_prototype(A_cam_o, A_F_proj_o, self.prototype_confidence_rate) # (n_class, n_channel)
        inv_A_prototype_t = self._get_prototype(inv_A_cam_t, inv_A_F_proj_t, self.prototype_confidence_rate) if self.use_inverse_affine else None # (n_class, n_channel)
        
        ## stop_gradient on pseudo pixel-level label and prototype
        pseudo_label_o, pseudo_label_t = tf.stop_gradient(pseudo_label_o), tf.stop_gradient(pseudo_label_t)
        A_pseudo_label_o, inv_A_pseudo_label_t = tf.stop_gradient(A_pseudo_label_o), tf.stop_gradient(inv_A_pseudo_label_t) if self.use_inverse_affine else None
        prototype_o, prototype_t = tf.stop_gradient(prototype_o), tf.stop_gradient(prototype_t)
        A_prototype_o, inv_A_prototype_t = tf.stop_gradient(A_prototype_o), tf.stop_gradient(inv_A_prototype_t) if self.use_inverse_affine else None
            
        ## flatten (batch_size, h, w) into (batch_size*h*w)
        F_proj_o = tf.reshape(F_proj_o, (-1, F_proj_o.shape[-1]))
        F_proj_t = tf.reshape(F_proj_t, (-1, F_proj_t.shape[-1]))
        A_F_proj_o = tf.reshape(A_F_proj_o, (-1, A_F_proj_o.shape[-1]))
        inv_A_F_proj_t = tf.reshape(inv_A_F_proj_t, (-1, inv_A_F_proj_t.shape[-1])) if self.use_inverse_affine else None
        pseudo_label_o = tf.reshape(pseudo_label_o, [-1])
        pseudo_label_t = tf.reshape(pseudo_label_t, [-1])
        A_pseudo_label_o = tf.reshape(A_pseudo_label_o, [-1])
        inv_A_pseudo_label_t = tf.reshape(inv_A_pseudo_label_t, [-1]) if self.use_inverse_affine else None
            
        ## compute contrastive loss
        L_cp = 0.5*(self._nce_loss(A_F_proj_o, A_pseudo_label_o, prototype_t) + self._nce_loss(F_proj_t, pseudo_label_t, A_prototype_o))
        L_cc = 0.5*(self._nce_loss(A_F_proj_o, pseudo_label_t, A_prototype_o) + self._nce_loss(F_proj_t, A_pseudo_label_o, prototype_t))
        if self.use_inverse_affine:
            L_cp += 0.5*(self._nce_loss(F_proj_o, pseudo_label_o, inv_A_prototype_t) + self._nce_loss(inv_A_F_proj_t, inv_A_pseudo_label_t, prototype_o))
            L_cc += 0.5*(self._nce_loss(F_proj_o, inv_A_pseudo_label_t, prototype_o) + self._nce_loss(inv_A_F_proj_t, pseudo_label_o, inv_A_prototype_t))
            L_cp, L_cc = 0.5*L_cp, 0.5*L_cc
        L_cp, L_cc = self.cp_contrast_coef*tf.reduce_mean(L_cp), self.cc_contrast_coef*tf.reduce_mean(L_cc)
            
        
        L_intra = 0.5*(self._nce_loss_hard_mining(F_proj_o, pseudo_label_o, prototype_o) + self._nce_loss_hard_mining(F_proj_t, pseudo_label_t, prototype_t))
        L_intra = self.intra_contrast_coef*L_intra # already scalar
            
        return (features + (F_proj_o, F_proj_t, A_F_proj_o, inv_A_F_proj_t, prototype_o, prototype_t, A_prototype_o, inv_A_prototype_t), 
                losses + (L_cp, L_cc, L_intra))
    
    def call(self, x, y=None, training=False):
        if training and y is not None:
            # apply affine
            affine_code, affine_args, inv_affine_args = self._ranomd_affine_code(tf.shape(x)[0])
            A_x = self.affiner.apply_affine(x, affine_code, affine_args)
            
            # get features from simsiam model
            _, F_o, cam_o, F_o_r, F_proj_o, pred_y_o = self.model(x, training=True)
            _, F_t, cam_t, F_t_r, F_proj_t, pred_y_t = self.model(A_x, training=True)
            
            # compute losses
            features, losses = self._loss(affine_code, affine_args, inv_affine_args, x, y, A_x, F_o, F_t, cam_o, cam_t, F_o_r, F_t_r, F_proj_o, F_proj_t, pred_y_o, pred_y_t)
            loss_cls, L_ER, L_ECR, loss_min, L_cp, L_cc, L_intra = losses   
            
            self.add_loss(L_ER)
            self.add_loss(L_ECR)
            self.add_loss(loss_min)
            self.add_loss(L_cp)
            self.add_loss(L_cc)
            self.add_loss(L_intra)
            
            L_all = sum(losses)
            self.add_metric(L_ER, name='L_ER')
            self.add_metric(L_ECR, name='L_ECR')
            self.add_metric(loss_min, name='L_min')
            self.add_metric(loss_cls, name='L_cls')
            self.add_metric(L_cp, name='L_cp')
            self.add_metric(L_cc, name='L_cc')
            self.add_metric(L_intra, name='L_intra')
            self.add_metric(L_all, name='L_all')
            
            pred_y_o = features[0]
            return pred_y_o
            
        else:
            return self.model(x)
        
    def get_config(self):
        return {**super(Pixel2Prototype, self).get_config(),
                "prototype_confidence_rate": self.prototype_confidence_rate,
                "nce_tau": self.nce_tau,
                "cp_contrast_coef": self.cp_contrast_coef,
                "cc_contrast_coef": self.cc_contrast_coef,
                "intra_contrast_coef": self.intra_contrast_coef,
                "hard_prototype_range": self.hard_prototype_range,
                "hard_pixel_range": self.hard_pixel_range,
                "background_threshold": self.background_threshold
               }
    
    @staticmethod
    def from_config(config):
        model = tf.keras.models.Model.from_config(config.pop("model"))
        image_input = model.inputs[0]
        feature_output, categorical_feature, cam_feature, refined_cam_feature, project_feature, classify_output = model.outputs
        p2p = Pixel2Prototype(image_input, feature_output, **config)
        p2p._build_models(image_input, feature_output, categorical_feature, 
                            cam_feature, refined_cam_feature, project_feature, 
                            classify_output)
        return p2p
    
    @staticmethod
    def load_model(filepath, **kwds):
        model = tf.keras.models.load_model(filepath)
        image_input = model.inputs[0]
        feature_output, categorical_feature, cam_feature, refined_cam_feature, project_feature, classify_output = model.outputs
        p2p = Pixel2Prototype(image_input, feature_output, **kwds)
        p2p._build_models(image_input, feature_output, categorical_feature, 
                            cam_feature, refined_cam_feature, project_feature, 
                            classify_output)
        return p2p