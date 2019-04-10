# CNN-on-CIFAR-10
## Simple implementation of CNN on CIFAR10
## 1 Algorithm and Python Code
A CNN has been created here for a training and testing framework using the cifar10 dataset using keras.
The convolutional neural network model created has 6 convolutional layers with activation and normaliza-
tion layers after every layer and a dropout an a maxpooling layer for the spatial downsampling, followed
my 2 dense neural layers.The final layer is flattened and connected to a dense layer of neurons with the
dimension equal to the number of classes. This model achieved 89.92% accuracy on test data and 92%
accuracy on training data. The model is built with reference to a model description here. However, the ad-
ditional changes improved the performance of the given model and improved the training and the testing
accuracy with negligible additional computational time.
### 1.1 Initialization
We start out by importing the CIFAR-10 dataset from keras and displaying one image(32X32) from each
of the 10 classes in the dataset as shown below:-

The images are then Z-score (mean-std) normalized because it results in similarly-ranged feature val-
ues and it ensures that the gradients don’t go out of control (need one global learning rate multiplier).
Finally we convert out training sets to binary encodings for easy classification.
### 1.2 Building the model
Next, we build our model of 6 convolution layers. The first 2 convolution layers have 32 classic convolution
filters of (3,3) dimensions. We increase the number of filters in the forthcoming layers but maintain the
same dimensions. We use "same" padding so that the image dimensions stay the same after the convolu-
tion operation. The Kernel regularizer allows to apply penalties on layer parameters during optimization.
These penalties are incorporated in the loss function that the network optimizes. This argument in con-
volutional layer is nothing but L2 regularisation of the weights. This penalizes peaky weights and makes

sure that all the inputs are considered. During gradient descent parameter update, the above L2 regu-
larization ultimately means that every weight is decayed linearly, that’s why called weight decay.An ’elu’

activation layer acts as the activation layer after every convolution layer followed by a batch normalization
layer. BatchNormalization normalizes the activation of the previous layer at each batch, i.e. applies a
transformation that maintains the mean activation close to 0 and the activation standard deviation close

to 1. It addresses the problem of internal covariate shift. It also acts as a regularizer, in some cases elim-
inating the need for Dropout. Batch Normalization achieves the same accuracy with fewer training steps

thus speeding up the training process. This layer is followed by a dropout and a maxpooling layer for
the spatial downsampling.Hence, overfitting is tackled. We pass the flattened output into 2 dense neural
layers of dimensions 64 and 32, each with relu activation functions. Finally the final output is connected
to a dense layer of neurons with the dimension equal to the number of classes and a softmax activation.
A description of the model used is given below:-
_____________________________________________________________
Layer (type) Output Shape Param
=================================================================
conv2d_1 (Conv2D) (None, 32, 32, 32) 896
_________________________________________________________________
activation_1 (Activation) (None, 32, 32, 32) 0
_________________________________________________________________
batch_normalization_1 (Batch (None, 32, 32, 32) 128
_________________________________________________________________
conv2d_2 (Conv2D) (None, 32, 32, 32) 9248
_________________________________________________________________
activation_2 (Activation) (None, 32, 32, 32) 0
_________________________________________________________________
batch_normalization_2 (Batch (None, 32, 32, 32) 128
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 16, 16, 32) 0
_________________________________________________________________
dropout_1 (Dropout) (None, 16, 16, 32) 0
_________________________________________________________________
conv2d_3 (Conv2D) (None, 16, 16, 64) 18496
_________________________________________________________________
activation_3 (Activation) (None, 16, 16, 64) 0
_________________________________________________________________
batch_normalization_3 (Batch (None, 16, 16, 64) 256
_________________________________________________________________
conv2d_4 (Conv2D) (None, 16, 16, 64) 36928
_________________________________________________________________
activation_4 (Activation) (None, 16, 16, 64) 0
_________________________________________________________________
batch_normalization_4 (Batch (None, 16, 16, 64) 256
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 8, 8, 64) 0
_________________________________________________________________
dropout_2 (Dropout) (None, 8, 8, 64) 0
_________________________________________________________________
conv2d_5 (Conv2D) (None, 8, 8, 128) 73856
_________________________________________________________________
activation_5 (Activation) (None, 8, 8, 128) 0
_________________________________________________________________
batch_normalization_5 (Batch (None, 8, 8, 128) 512
_________________________________________________________________
conv2d_6 (Conv2D) (None, 8, 8, 128) 147584
_________________________________________________________________
activation_6 (Activation) (None, 8, 8, 128) 0
_________________________________________________________________
batch_normalization_6 (Batch (None, 8, 8, 128) 512
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 4, 4, 128) 0
_________________________________________________________________
dropout_3 (Dropout) (None, 4, 4, 128) 0
_________________________________________________________________
flatten_1 (Flatten) (None, 2048) 0
_________________________________________________________________
dense_1 (Dense) (None, 64) 131136
_________________________________________________________________
dense_2 (Dense) (None, 32) 2080
_________________________________________________________________
dense_3 (Dense) (None, 10) 330
=================================================================
Total params: 422,346 Trainable params: 421,450 Non-trainable params: 896

### 1.3 Image Augmentation

We use a ImageDataGenerator class generate batches of tensor image data with real-time data augmenta-
tion. The data will be looped over (in batches) indefinitely. The image data is generated by transforming

the actual training images by rotation, crop, shifts, shear, zoom, flip, reflection, normalization etc.
### 1.4 Optimizer and Loss
For the back-propagation,Adam Optimizer is used as it gave better results in less computation time than
gradient descent. Categorical crossentropy is used for the loss calculation.
### 1.5 Operation
The built network is trained by using random batches of 64 samples running the algorithm for a total
of 140 epochs.The training performance thus obtained is 92% and the testing performance achieved is
almost 90% . The testing is done by taking batch sizes of 128.

## 2 Training and Testing Performance
The output from the console on running the above code for the testing performance is mentioned as
follows:-
Epoch 133/140
781/781 [==============================] - 41s 52ms/step - loss: 0.3015 - acc: 0.9154
- val_loss: 0.4045 - val_acc: 0.8970
Epoch 134/140

781/781 [==============================] - 40s 51ms/step - loss: 0.3025 - acc: 0.9162
- val_loss: 0.4053 - val_acc: 0.8933
Epoch 135/140
781/781 [==============================] - 52s 66ms/step - loss: 0.2986 - acc: 0.9170
- val_loss: 0.4109 - val_acc: 0.8934
Epoch 136/140
781/781 [==============================] - 51s 66ms/step - loss: 0.2990 - acc: 0.9163
- val_loss: 0.4070 - val_acc: 0.8929
Epoch 137/140
781/781 [==============================] - 52s 66ms/step - loss: 0.3016 - acc: 0.9171
- val_loss: 0.4162 - val_acc: 0.8914
Epoch 138/140
781/781 [==============================] - 44s 56ms/step - loss: 0.3000 - acc: 0.9168
- val_loss: 0.4008 - val_acc: 0.8948
Epoch 139/140
781/781 [==============================] - 43s 55ms/step - loss: 0.2949 - acc: 0.9191
- val_loss: 0.3934 - val_acc: 0.8975
Epoch 140/140
781/781 [==============================] - 42s 54ms/step - loss: 0.2976 - acc: 0.9169
- val_loss: 0.3859 - val_acc: 0.8992
10000/10000 [==============================] - 1s 125us/step
We can observe that an accuracy of 91.69% is achieved at the end of a 140 iterations with random sets of
batch-sizes of 64.Similarly, we note a testing accuracy of 89.92% from the following output value using
the CIFAR-10 testing samples: Test result: 89.920 loss: 0.386
## 3 References
• https://appliedmachinelearning.blog/2018/03/24/
• https://www.kaggle.com/c/cifar-10/discussion/40237
• https://www.coursera.org/learn/convolutional-neural-networks/
