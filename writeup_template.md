# **Traffic Sign Recognition** 

## Writeup - Sara Collins
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[train_dist]: ./writeup_pics/train_class_dist.png "Training target distribution"
[test_dist]: ./writeup_pics/test_class_dist.png "Testing target distribution"
[valid_dist]: ./writeup_pics/valid_class_dist.png "Validation target distribution"
[total_dist]: ./writeup_pics/total_class_dist.png "Total target value distribution"
[original_preprocess]: ./writeup_pics/original_preprocess.png "Original preprocessing image"
[gray_scale_only]: ./writeup_pics/gray_scale_only.png "Grayscale preprocessing example"
[normal_preprocess]: ./writeup_pics/normal_preprocess.png "Normalized image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  


### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

 [My Project Notebook](CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier_SEC.ipynb)
 
### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used functions found in the pickled pandas DataFrame object that held the images to calculate summary statistics of the traffic
signs data set:

* The size of training set is: 34,799 examples
* The size of the validation set is: 4,410 examples
* The size of test set is: 12,360 examples
* The shape of a traffic sign image is: 32 x 32 x 3 (32x32 pixels for height and width, with three color channels)
* The number of unique classes/labels in the data set is: 43

#### 2. Include an exploratory visualization of the dataset.

In initially exploring the dataset, knowing how the distribution of values of a target variable lies is very important. 
Not only do machine learning algorithms often take distributions into account, but it is also important to check to make 
sure any holdout sets for validation and testing give a good approximation of the total data distribution. For example,
if this dataset total at any given point in time has roughly 50% stop signs and the remainder of the signs distributed
pretty evenly, a validation set that only has 5% of stop signs would likely be problematic. Also, depending on the
algorithms used, it may be necessary to downsample or oversample some classes in order to ensure the algorithm will train 
correctly. 

For brevity's sake, I will include only images I used for the target variable here. In non-image datasets, I often use 
the Seaborn package's 'factorplot' to do a visual distribution and correlation check on ALL variables in the dataset.

Here is the distribution of the entire dataset, including target variables from the train, test, and validation sets all 
together so the best approximation of the total distribution of the entire population can be seen (in real life, I tend to 
do this from historical data that is completely separate from the validation and testing sets, but since this dataset is 
not part of a larger data ecosystem, I just combined the original three sets for exploration). 
Also note that although the sign classes are numeric in this case, this is a distibution of sign class counts, not the 
integer value of the signs' class. 
#### Entire dataset:
![total distribution][total_dist]

Now, to compare, here are the distributions of the similar subsets. 
#### Train Dataset: 
![training distribution][train_dist]  
  
  
#### Validation Dataset:  
![validation distribution][valid_dist]  
  
  
#### Test Dataset:
![test distribution][test_dist]
  
 

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

To preprocess the images prior to training a convolutional model on it, I did a few things. 
While grayscale was used as an option in other labs for this course, I noticed during training that it did not 
appear to yield as good of an accuracy score as just leaving 3 dimensions for color in the images, so I removed this 
step. Thinking critically about this, while markings on signs are usually very distinct, the color of a traffic sign also usually conveys important information to drivers, particularly for sign groupings, and presence or absence of certain colors might help a classifier be more certain of the probability of a sign. 

**Original Image:**  
![alt_text][original_preprocess]

**Gray-scale applied to original image:**  
![alt text][gray_scale_only]

Instead of applying gray-scale since it appeared to make the accuracy score worse, I just normalized the image data to elimnate outlier pixel values and to make sure the pixel values have a zero mean and equal variance. 

**Normalization applied to original image:**  
![alt_text][normal_preprocess]
  
  

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I used the LeNet model as the basis for the traffic sign classification model. The structure was as follows:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image - Normalized, but NOT grayscale (hence the 3 color channels)	| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16 |
| RELU     |            |
| Max pooling		| 2x2 stride, outputs 5x5x16  			|
| Flatten | Compress to 1D array of 400       		|
|	Fully Connected					|		Output 120										|
|	RELU					|												|
| Fully Connected     | Output 84  |
| RELU     |            |
| Fully Connected     | Output 43, for the number of classes  |

 
#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I cycled through various hyperparameters combinations and preprocessing steps.
The best results I obtained included an Adam optimizer, used 40 epochs, had a batch size of 128, and
a learning rate of .0001.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
(Training: Not printed during last run)
* validation set accuracy of 64.9%
* test set accuracy of 6.7%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? 
I tried the LeNet architecture as it was during the lecture that introduced it. In the initial trial,
I tried using a grayscale transformation on the image as well. 
* What were some problems with the initial architecture?
Based on the validation metrics coming out during training, I ended up readjusting the batch size and 
noticed that it was higher without the grayscaled images. 
* How was the architecture adjusted and why was it adjusted? 
I attempted to add a convolutional layer to help what appeared to be underfitting of the model, and even just adding a 
single additional convolutional layer appeared to make the model overfit badly. From there I also adjusted preprocessing 
and hyperparameters, but it was still overfitting, so I took the additional convolutional layer back out even though it 
appears to have performed poorly on the test set. 
* Which parameters were tuned? How were they adjusted and why?
I adjusted batch size so it always was a double of the next one higher- 128 appeared to give the best results given the rest of the setup and hyperparameters. The number of epochs that seemed to run best was 40. I tried up to 80, but the model appeared to converge at around 42-ish, so I noted that I had the best results with 40 epochs. Learning rate also seemed to have the validation metric 'hop around' too much during training at the original .001, so I decreased it to .0001 with what appeared to be better results. 
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem?
Convolutional layers are important because they help make the best sense of the model. Pooling layers are also important to try to remove additional 'noisy' features from the model. Activation functions are important to convert to nonlinearity and also keep the gradients from shrinking and/or vanishing. 
If a well known architecture was chosen:
* What architecture was chosen?  
LeNet
* Why did you believe it would be relevant to the traffic sign application?
It had a few convolutional layers built in already and is used in other problems for identifying details, so I figured it would be a good starting point for this problem as well. 
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?  
The training and validation sets appeared to be working well together during training itself, by not diverging and moving/converging in an appropriate fashion. However, performance on this particular testing set would indicate that severe overfitting occurred. 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are fifteen more German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


