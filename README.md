# SEAM
The implementation of Self-supervised Equivariant Attention Mechanism for Weakly Supervised Semantic Segmentaion.

# Usage
## SEAM
Define backbone of CNN:
```python
img_input = tf.keras.layers.Input((28, 28, 1))
x = tf.keras.layers.Conv2D(64, 3, padding="same")(img_input)
x = tf.keras.layers.BatchNormalization()(x)
x0 = tf.keras.layers.Activation("relu")(x)
x = tf.keras.layers.Conv2D(128, 3, padding="same")(x0)
```

Create SEAM model:
```python
seam = SEAM(img_input, x, classes=10)
```

Train model:
```python
seam.compile(tf.keras.optimizers.Adam(learning_rate=1e-4), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
seam.fit(X_train, y_train)
```

Prediction:
```python
pred_y = seam.predict(X_test)
```


Save and load model:
```python
seam.save("./model.h5")
seam1 = SEAM.load("./model.h5", classes=10)
```


## Affine
Read image:

```python
I = cv2.imread("./img.jpg").astype(np.float32) / 255
I = np.expand_dims(I, 0) # [batch_size, height, width, n_channel]
```

Generate affine code and arguments:
```python
affine = Affine()
code, args = affine.random_affine_code()
inv_args = affine.inverse_affine_code(code, args)
```

Apply affine:
```python
A_I = affine.apply_affine(I, code, args)     # transformed image
R_I = affine.apply_affine(I, code, inv_args) # inverse transformed image
```


# References
1. [Self-supervised Equivariant Attention Mechanism for Weakly Supervised Semantic Segmentaion](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Self-Supervised_Equivariant_Attention_Mechanism_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2020_paper.pdf)
2. [Non-local Neural Networks](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.pdf)
3. [https://github.com/YudeWang/SEAM](https://github.com/YudeWang/SEAM) 
