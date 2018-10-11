# carvana-image-masking-challenge-michael


[//]: # (Image References)

[image1]: ./image/data.png "Data"
[image2]: ./image/data_aug.png "Data Aug"
[image3]: ./image/test_accuracy.png "Test Accuracy"

Overview
---
This repository if for the kaggle competition with carvana-image-masking-challenge. It contains following file:

* train.ipynb (Script used for data cleaning , splitting , build the model and training)

The task is that we need to train one model to help removing the background color from the car images in different angles. The traditional CNN network is used in classification tasks.

But the label of this project is to find the contour of car so that we need different CNN network. After a bit research, I find that U-Net CNN network is fit for this task and U-Net is specially used in biomedical images segmentation.

U-Net is not only can do the classification job but it also can do the localization job. U-Net is a full convolutional network. It uses contracting network which contains multiple layers to extract the feature maps and increase the image depth.  In the up-sampling part, the high resolution feature from contracting network will be combined to the features. Hence the network is able to propagate the image context information successive layers. 


Model Architecture Description
---

The are several architecture we can choose for this segmentation task. For example, U-Net, DeepLab, SegNet and ResNet etc. In this repo, I choosed U-Net as it requires fewer training images and also the training speed is also fast. The details of U-Net can be found here: https://arxiv.org/pdf/1505.04597.pdf  

Due to the GPU memory limitation, I have resized the image to 128*128 so that the input of image size will be 128*128*3. Here is the model summary I used in this project. In the high level, it includes one 'down' and 'up' network. Inside the 'down' layer, it includes the Conv2D, Activation, Max Pooling and batch normalization. In the 'up' layer, it includes the UpSampling2D, Concatenate the relative 'down' layer, Conv2D, Activation, and batch normalization etc.

__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 128, 128, 3)  0                                            
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 128, 128, 64) 1792        input_1[0][0]                    
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 128, 128, 64) 256         conv2d_1[0][0]                   
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 128, 128, 64) 0           batch_normalization_1[0][0]      
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 128, 128, 64) 36928       activation_1[0][0]               
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 128, 128, 64) 256         conv2d_2[0][0]                   
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 128, 128, 64) 0           batch_normalization_2[0][0]      
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 64, 64, 64)   0           activation_2[0][0]               
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 64, 64, 128)  73856       max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 64, 64, 128)  512         conv2d_3[0][0]                   
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 64, 64, 128)  0           batch_normalization_3[0][0]      
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 64, 64, 128)  147584      activation_3[0][0]               
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 64, 64, 128)  512         conv2d_4[0][0]                   
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 64, 64, 128)  0           batch_normalization_4[0][0]      
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 32, 32, 128)  0           activation_4[0][0]               
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 32, 32, 256)  295168      max_pooling2d_2[0][0]            
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 32, 32, 256)  1024        conv2d_5[0][0]                   
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 32, 32, 256)  0           batch_normalization_5[0][0]      
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 32, 32, 256)  590080      activation_5[0][0]               
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 32, 32, 256)  1024        conv2d_6[0][0]                   
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 32, 32, 256)  0           batch_normalization_6[0][0]      
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 16, 16, 256)  0           activation_6[0][0]               
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 16, 16, 512)  1180160     max_pooling2d_3[0][0]            
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 16, 16, 512)  2048        conv2d_7[0][0]                   
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 16, 16, 512)  0           batch_normalization_7[0][0]      
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 16, 16, 512)  2359808     activation_7[0][0]               
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 16, 16, 512)  2048        conv2d_8[0][0]                   
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 16, 16, 512)  0           batch_normalization_8[0][0]      
__________________________________________________________________________________________________
max_pooling2d_4 (MaxPooling2D)  (None, 8, 8, 512)    0           activation_8[0][0]               
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 8, 8, 1024)   4719616     max_pooling2d_4[0][0]            
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 8, 8, 1024)   4096        conv2d_9[0][0]                   
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 8, 8, 1024)   0           batch_normalization_9[0][0]      
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 8, 8, 1024)   9438208     activation_9[0][0]               
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 8, 8, 1024)   4096        conv2d_10[0][0]                  
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 8, 8, 1024)   0           batch_normalization_10[0][0]     
__________________________________________________________________________________________________
up_sampling2d_1 (UpSampling2D)  (None, 16, 16, 1024) 0           activation_10[0][0]              
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 16, 16, 1536) 0           activation_8[0][0]               
                                                                 up_sampling2d_1[0][0]            
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 16, 16, 512)  7078400     concatenate_1[0][0]              
__________________________________________________________________________________________________
batch_normalization_11 (BatchNo (None, 16, 16, 512)  2048        conv2d_11[0][0]                  
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 16, 16, 512)  0           batch_normalization_11[0][0]     
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 16, 16, 512)  2359808     activation_11[0][0]              
__________________________________________________________________________________________________
batch_normalization_12 (BatchNo (None, 16, 16, 512)  2048        conv2d_12[0][0]                  
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 16, 16, 512)  0           batch_normalization_12[0][0]     
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 16, 16, 512)  2359808     activation_12[0][0]              
__________________________________________________________________________________________________
batch_normalization_13 (BatchNo (None, 16, 16, 512)  2048        conv2d_13[0][0]                  
__________________________________________________________________________________________________
activation_13 (Activation)      (None, 16, 16, 512)  0           batch_normalization_13[0][0]     
__________________________________________________________________________________________________
up_sampling2d_2 (UpSampling2D)  (None, 32, 32, 512)  0           activation_13[0][0]              
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 32, 32, 768)  0           activation_6[0][0]               
                                                                 up_sampling2d_2[0][0]            
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 32, 32, 256)  1769728     concatenate_2[0][0]              
__________________________________________________________________________________________________
batch_normalization_14 (BatchNo (None, 32, 32, 256)  1024        conv2d_14[0][0]                  
__________________________________________________________________________________________________
activation_14 (Activation)      (None, 32, 32, 256)  0           batch_normalization_14[0][0]     
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 32, 32, 256)  590080      activation_14[0][0]              
__________________________________________________________________________________________________
batch_normalization_15 (BatchNo (None, 32, 32, 256)  1024        conv2d_15[0][0]                  
__________________________________________________________________________________________________
activation_15 (Activation)      (None, 32, 32, 256)  0           batch_normalization_15[0][0]     
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 32, 32, 256)  590080      activation_15[0][0]              
__________________________________________________________________________________________________
batch_normalization_16 (BatchNo (None, 32, 32, 256)  1024        conv2d_16[0][0]                  
__________________________________________________________________________________________________
activation_16 (Activation)      (None, 32, 32, 256)  0           batch_normalization_16[0][0]     
__________________________________________________________________________________________________
up_sampling2d_3 (UpSampling2D)  (None, 64, 64, 256)  0           activation_16[0][0]              
__________________________________________________________________________________________________
concatenate_3 (Concatenate)     (None, 64, 64, 384)  0           activation_4[0][0]               
                                                                 up_sampling2d_3[0][0]            
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 64, 64, 128)  442496      concatenate_3[0][0]              
__________________________________________________________________________________________________
batch_normalization_17 (BatchNo (None, 64, 64, 128)  512         conv2d_17[0][0]                  
__________________________________________________________________________________________________
activation_17 (Activation)      (None, 64, 64, 128)  0           batch_normalization_17[0][0]     
__________________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, 64, 64, 128)  147584      activation_17[0][0]              
__________________________________________________________________________________________________
batch_normalization_18 (BatchNo (None, 64, 64, 128)  512         conv2d_18[0][0]                  
__________________________________________________________________________________________________
activation_18 (Activation)      (None, 64, 64, 128)  0           batch_normalization_18[0][0]     
__________________________________________________________________________________________________
conv2d_19 (Conv2D)              (None, 64, 64, 128)  147584      activation_18[0][0]              
__________________________________________________________________________________________________
batch_normalization_19 (BatchNo (None, 64, 64, 128)  512         conv2d_19[0][0]                  
__________________________________________________________________________________________________
activation_19 (Activation)      (None, 64, 64, 128)  0           batch_normalization_19[0][0]     
__________________________________________________________________________________________________
up_sampling2d_4 (UpSampling2D)  (None, 128, 128, 128 0           activation_19[0][0]              
__________________________________________________________________________________________________
concatenate_4 (Concatenate)     (None, 128, 128, 192 0           activation_2[0][0]               
                                                                 up_sampling2d_4[0][0]            
__________________________________________________________________________________________________
conv2d_20 (Conv2D)              (None, 128, 128, 64) 110656      concatenate_4[0][0]              
__________________________________________________________________________________________________
batch_normalization_20 (BatchNo (None, 128, 128, 64) 256         conv2d_20[0][0]                  
__________________________________________________________________________________________________
activation_20 (Activation)      (None, 128, 128, 64) 0           batch_normalization_20[0][0]     
__________________________________________________________________________________________________
conv2d_21 (Conv2D)              (None, 128, 128, 64) 36928       activation_20[0][0]              
__________________________________________________________________________________________________
batch_normalization_21 (BatchNo (None, 128, 128, 64) 256         conv2d_21[0][0]                  
__________________________________________________________________________________________________
activation_21 (Activation)      (None, 128, 128, 64) 0           batch_normalization_21[0][0]     
__________________________________________________________________________________________________
conv2d_22 (Conv2D)              (None, 128, 128, 64) 36928       activation_21[0][0]              
__________________________________________________________________________________________________
batch_normalization_22 (BatchNo (None, 128, 128, 64) 256         conv2d_22[0][0]                  
__________________________________________________________________________________________________
activation_22 (Activation)      (None, 128, 128, 64) 0           batch_normalization_22[0][0]     
__________________________________________________________________________________________________
conv2d_23 (Conv2D)              (None, 128, 128, 1)  65          activation_22[0][0]              
__________________________________________________________________________________________________

Total params: 34,540,737

Trainable params: 34,527,041

Non-trainable params: 13,696


Training data preparation
---

The training data has been given by the project organizer. First we need to understand the basic info from our training data and it's like following:

Number of training examples = 4071
Number of valid examples = 1018
Number of testing examples = 100064
Image data shape = (1280, 1918, 3)

Also note that the label of the target image is just some values with 0 and 1 only.

The training data image, the label image and the test image can be visualized like below:

![training data][image1]
 


Data augmentation
---

As the training data is limited so that we need to do data augmentation to increase the data volume. Following methods have been implemented:

* Image flip
* Images rotation with -8 to 8 degree range.

Here is the image flip and image rotation example for test image:

![data augmentation][image2]


Keras Generator
---
If we pre-process all images directly, it will consume a lot memory and will be a limitation if we need tons of images ot train. The generator is here to help. However, to understan how Keras generator works is not easy as the generator is not that straightforward.

Basically the generator used the static variable internally and it will remember the last execution status and will continue from the status from last run. So if we pass the training samples via model.fit_generator, the generator will yield one array with length batch_size. Let's say we have 10 training examples and batch size is 2, then it will yield 5 times to go though all data samples. Based on this, we should set the samples_per_epoch= len(train_samples)/batch_size+1, and it will make training more faster and have more chances to get a better trained model and prevent the over-fitting.

Test Accuracy
---

Based on the training model, I predicated the test image and could find that there some false positives and need to to more data augmentation to increase the training data size and also need to try different training model to compare the result. Here is the predicated test image label based on the training model.

 
![test accuracy][image3]


Summary
---

Though I have some experience in CNN network, I am quite new to image segmentation. I enjoyed training a such complex network and glad to see the test accuracy is increasing. I used Google colab free GPU to train this model and feel that it's a bit slow. I am thinking need to implement the multiple threading of read images to release the power of multi-core CPU, with that I would have more time to optimize the network and add more data augmentation methods in the image processing phase.

The current test accuracy is at around 0.983 and I believe it's due to the image size was minimized to 128*128. I think I can get a better result with a higher resolution image in training model.

Appreciate sincerely for Sankha Mukherjee to give an good opportunity to implement it and I feel that I can implement what I learned into the real world project. 
