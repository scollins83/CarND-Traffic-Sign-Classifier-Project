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

[fifteen_images]: fifteen_image_test_set/fifteen_additional_pictures.png "Fifteen Original Signs"
[fifteen_normed_images]: fifteen_image_test_set/normed_fifteen_additional_pictures.png "Fifteen Normalized Signs"
[train_dist]: ./writeup_pics/train_class_dist.png "Training target distribution"
[test_dist]: ./writeup_pics/test_class_dist.png "Testing target distribution"
[valid_dist]: ./writeup_pics/valid_class_dist.png "Validation target distribution"
[total_dist]: ./writeup_pics/total_class_dist.png "Total target value distribution"
[original_preprocess]: ./writeup_pics/original_preprocess.png "Original Example"
[gray_scale_only]: ./writeup_pics/gray_scale_only.png "Grayscale Example"
[normal_preprocess]: ./writeup_pics/normal_preprocess.png "Normalized Example"
[normal_grayscale_preprocess]: ./writeup_pics/normalized_grayscale_preprocess.png "Grayscaled and then Normalized Example"

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

To preprocess the images prior to training a convolutional model on it, I did a few things. I will walk through that process, showing some examples. 

**Original Image:**  
![alt_text][original_preprocess]

First, I converted the images to grayscale. Looking visually at the image afterward, this can help simplify the features, and may make it simpler for the model to learn about shapes of signs and characters/icons on the sign. 

**Gray-scale applied to original image:**  
![alt text][gray_scale_only]

Next, I normalized the grayscaled image, to bring values within a range relative to one another and making it easier for the model to learn necessary patterns from the pixel values in general. 

**Normalization applied to grayscaled image, and also what the same normalization function looks like when applied to the original 3-channel color image:**  
![alt_text][normal_grayscale_preprocess] vs. ![alt_text][normal_preprocess]
  
Lastly, I had to reshape the images to make sure the single channel for the grayscale image was explicitly noted, in order to be able to feed it to the tensorflow input layers. This didn't change the image visually.  

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I used the LeNet model as the basis for the traffic sign classification model. The structure was as follows:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 images - Normalized and Grayscaled	| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 |
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
a learning rate of .001. Early stopping could perhaps have been utilized, but it tended to still cycle around a bit at 40 epochs. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 93.9%
* test set accuracy of 91.9%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? 
I tried the LeNet architecture as it was during the lecture that introduced it.
* What were some problems with the initial architecture?
Based on the validation metrics coming out during training, I ended up readjusting the batch size and 
also tried different preprocessing approaches, seeing what would happen with and without normalization. In the meantime, I had to adjust the inputs and reshape the images because using the CV2 grayscale conversion method didn't yield the right explicit data shape for tensorflow to use. 
* How was the architecture adjusted and why was it adjusted? 
I attempted to add a convolutional layer to help what initially appeared to be underfitting of the model during some of my earlier trials, and even just adding a single additional convolutional layer appeared to make the model overfit badly. From there I also adjusted preprocessing and hyperparameters, but it was still overfitting, so I took the additional convolutional layer back out and my output validation metrics were much better. 
* Which parameters were tuned? How were they adjusted and why?
I adjusted batch size so it always was a double of the next one higher- 128 appeared to give the best results given the rest of the setup and hyperparameters. The number of epochs that seemed to run best was 40. I tried up to 80, but the model appeared to converge at around 42-ish, so I noted that I had the best results with 40 epochs. Learning rate also seemed to have the validation metric 'hop around' too much during training at the original .01, and moved quite slowly on .0001, and 0.001 appeared to have the best results. 
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem?
Convolutional layers are important because they help make the best sense of the model. Pooling layers are also important to try to remove additional 'noisy' features from the model. Activation functions are important to convert to nonlinearity and also keep the gradients from shrinking and/or vanishing. 
If a well known architecture was chosen:
* What architecture was chosen?  
LeNet
* Why did you believe it would be relevant to the traffic sign application?
It had a few convolutional layers built in already and is used in other problems for identifying details, so I figured it would be a good starting point for this problem as well and ended up performing sufficiently for the task at hand. 
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?  
The training and validation sets appeared to be working well together during training itself, by not diverging and also moving/converging in an appropriate fashion. The test set's accuracy is slightly lower than the training set, but not significantly so, and thus it appears that the model is not reasonably overfitting. 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are fifteen more German traffic signs that I found on the web:

![alt text][fifteen_images]  
  
Numbering the images as they are labeled:  
0 _  1 _  2   
3 _  4 _  5  ....

Here are the same images, after preprocessing:
![alt text][fifteen_normed_images]

Looking at the preprocessed images, all images except for 1, 2, 8, and 13 intuitively look as though they should readily be easy to recognize by the classifier. 
Things that may make the other images difficult to classify:  
1: Background building makes lines that might be easy to mix with the sign itself.  
2: Shadows/leaves across the '70' in the speed limit sign make it difficult to tell what that number is.  
8: The bicycle icon in the sign is a little blurry, and it's not a very common shape. It could be easy to confuse with similarly shaped signs.  
13: Like image 8, the animal icon in the sign is a little blurry, and it's not a very common shape. It could be easy to confuse with similarly shaped signs. 


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0 - Roundabout mandatory     		|  Priority road with 99% certainty (but roundabout mandatory was the 2nd listing)								| 
| 1 - Road work     			| Road work 										|
| 2 - Speed limit (70km/h)					| Speed limit (70km/h)										|
| 3 - Road work	      		| Road work					 				|
| 4 - Stop	| Stop      							|
| 5 - Speed limit (30km/h)    		| Speed limit (30km/h)  									| 
| 6 - No entry     			| No entry 										|
| 7 - Speed limit (80km/h)		| Speed limit (80km/h)									|
| 8 - Bicycles crossing    	| Bicycles crossing					|
| 9 - Stop | Stop    				|
| 10 - General caution    		| General caution  									| 
| 11 - Stop     			| Stop 										|
| 12 - Turn right ahead				| Turn right ahead											|
| 13 - Wild animals crossing      		| Wild animals crossing		 				|
| 14 - Speed limit (70km/h)		| Speed limit (70km/h)     							|


The model was able to correctly guess 14 of the 15 traffic signs, which gives an accuracy of **93.3%**. This is closer to the training validation of 93.9%, so this indicates that the model generalized well to these images. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were


**Record: 0**
 * Probability #1: 
      * Class Label: Priority road
      * Probability Value - 0.996679

 * Probability #2: 
      * Class Label: Roundabout mandatory
      * Probability Value - 0.00323869

 * Probability #3: 
      * Class Label: Speed limit (100km/h)
      * Probability Value - 8.23032e-05

 * Probability #4: 
      * Class Label: End of all speed and passing limits
      * Probability Value - 1.66307e-07

 * Probability #5: 
      * Class Label: Speed limit (120km/h)
      * Probability Value - 1.70323e-08




**Record: 1**
 * Probability #1: 
      * Class Label: Road work
      * Probability Value - 1.0

 * Probability #2: 
      * Class Label: Road narrows on the right
      * Probability Value - 6.94267e-17

 * Probability #3: 
      * Class Label: Speed limit (80km/h)
      * Probability Value - 5.24238e-17

 * Probability #4: 
      * Class Label: Dangerous curve to the right
      * Probability Value - 2.23018e-18

 * Probability #5: 
      * Class Label: Bumpy road
      * Probability Value - 4.83682e-19




**Record: 2**
* Probability #1: 
      * Class Label: Speed limit (70km/h)
      * Probability Value - 0.972135

 * Probability #2: 
      * Class Label: Speed limit (30km/h)
      * Probability Value - 0.0278651

 * Probability #3: 
      * Class Label: Speed limit (80km/h)
      * Probability Value - 2.50644e-12

 * Probability #4: 
      * Class Label: Road work
      * Probability Value - 5.78686e-18

 * Probability #5: 
      * Class Label: Speed limit (50km/h)
      * Probability Value - 2.23039e-18




**Record: 3**
* Probability #1: 
      * Class Label: Road work
      * Probability Value - 0.999999

 * Probability #2:
      * Class Label: Speed limit (80km/h)
      * Probability Value - 9.81208e-07

 * Probability #3: 
      * Class Label: Dangerous curve to the right
      * Probability Value - 2.42511e-08

 * Probability #4: 
      * Class Label: No passing for vehicles over 3.5 metric tons
      * Probability Value - 6.14101e-09

 * Probability #5: 
      * Class Label: Bumpy road
      * Probability Value - 2.5833e-09




**Record: 4**
* Probability #1: 
      * Class Label: Stop
      * Probability Value - 1.0

 * Probability #2: 
      * Class Label: No vehicles
      * Probability Value - 1.06898e-12

 * Probability #3: 
      * Class Label: Keep right
      * Probability Value - 1.61259e-13

 * Probability #4: 
      * Class Label: Speed limit (70km/h)
      * Probability Value - 4.26623e-14

 * Probability #5: 
      * Class Label: Speed limit (30km/h)
      * Probability Value - 3.94399e-14




**Record: 5**
* Probability #1: 
      * Class Label: Speed limit (30km/h)
      * Probability Value - 1.0

 * Probability #2: 
      * Class Label: Speed limit (70km/h)
      * Probability Value - 2.4669e-11

 * Probability #3: 
      * Class Label: Speed limit (20km/h)
      * Probability Value - 2.3222e-11

 * Probability #4: 
      * Class Label: Speed limit (50km/h)
      * Probability Value - 2.99194e-20

 * Probability #5: 
      * Class Label: Speed limit (80km/h)
      * Probability Value - 6.56019e-22




**Record: 6**
* Probability #1: 
      * Class Label: No entry
      * Probability Value - 1.0

 * Probability #2: 
      * Class Label: Stop
      * Probability Value - 2.94668e-16

 * Probability #3: 
      * Class Label: No passing
      * Probability Value - 1.82164e-22

 * Probability #4: 
      * Class Label: Keep left
      * Probability Value - 4.3587e-27

 * Probability #5: 
      * Class Label: Speed limit (70km/h)
      * Probability Value - 2.85273e-29




**Record: 7**
* Probability #1: 
      * Class Label: Speed limit (80km/h)
      * Probability Value - 0.999981

 * Probability #2: 
      * Class Label: Speed limit (50km/h)
      * Probability Value - 1.89584e-05

 * Probability #3: 
      * Class Label: Speed limit (30km/h)
      * Probability Value - 1.37245e-07

 * Probability #4: 
      * Class Label: Speed limit (100km/h)
      * Probability Value - 1.18157e-07

 * Probability #5: 
      * Class Label: Speed limit (60km/h)
      * Probability Value - 1.75002e-09




**Record: 8**
* Probability #1: 
      * Class Label: Bicycles crossing
      * Probability Value - 1.0

 * Probability #2: 
      * Class Label: Children crossing
      * Probability Value - 1.17513e-08

 * Probability #3: 
      * Class Label: Road narrows on the right
      * Probability Value - 3.67772e-09

 * Probability #4: 
      * Class Label: Road work
      * Probability Value - 3.48906e-10

 * Probability #5: 
      * Class Label: Bumpy road
      * Probability Value - 1.73925e-10




**Record: 9**
* Probability #1: 
      * Class Label: Stop
      * Probability Value - 1.0

 * Probability #2: 
      * Class Label: Priority road
      * Probability Value - 1.43776e-13

 * Probability #3: 
      * Class Label: Speed limit (30km/h)
      * Probability Value - 2.28386e-16

 * Probability #4: 
      * Class Label: Turn right ahead
      * Probability Value - 7.86251e-17

 * Probability #5: 
      * Class Label: Keep right
      * Probability Value - 7.34148e-17




**Record: 10**
 * Probability #1: 
      * Class Label: General caution
      * Probability Value - 1.0

 * Probability #2: 
      * Class Label: Pedestrians
      * Probability Value - 1.98359e-16

 * Probability #3: 
      * Class Label: Right-of-way at the next intersection
      * Probability Value - 2.48454e-20

 * Probability #4: 
      * Class Label: Traffic signals
      * Probability Value - 1.4698e-28

 * Probability #5: 
      * Class Label: Children crossing
      * Probability Value - 3.84731e-30




**Record: 11**
 * Probability #1: 
      * Class Label: Stop
      * Probability Value - 1.0

 * Probability #2: 
      * Class Label: Speed limit (70km/h)
      * Probability Value - 2.54598e-17

 * Probability #3: 
      * Class Label: Speed limit (30km/h)
      * Probability Value - 7.63573e-18

 * Probability #4: 
      * Class Label: No vehicles
      * Probability Value - 1.21613e-19

 * Probability #5: 
      * Class Label: Keep right
      * Probability Value - 1.18259e-19




**Record: 12**
 * Probability #1: 
      * Class Label: Turn right ahead
      * Probability Value - 1.0

 * Probability #2: 
      * Class Label: Go straight or left
      * Probability Value - 5.13152e-09

 * Probability #3: 
      * Class Label: Keep left
      * Probability Value - 4.63151e-10

 * Probability #4: 
      * Class Label: Roundabout mandatory
      * Probability Value - 1.64309e-13

 * Probability #5: 
      * Class Label: No passing
      * Probability Value - 5.28265e-16




**Record: 13**
 * Probability #1: 
      * Class Label: Wild animals crossing
      * Probability Value - 1.0

 * Probability #2: 
      * Class Label: Double curve
      * Probability Value - 1.7053e-07

 * Probability #3: 
      * Class Label: Road narrows on the right
      * Probability Value - 5.95641e-11

 * Probability #4: 
      * Class Label: Road work
      * Probability Value - 2.79197e-11

 * Probability #5: 
      * Class Label: Beware of ice/snow
      * Probability Value - 6.83629e-14




**Record: 14**
 * Probability #1: 
      * Class Label: Speed limit (70km/h)
      * Probability Value - 1.0

 * Probability #2: 
      * Class Label: Speed limit (20km/h)
      * Probability Value - 6.03635e-12

 * Probability #3: 
      * Class Label: Speed limit (30km/h)
      * Probability Value - 1.4846e-15

 * Probability #4: 
      * Class Label: Speed limit (120km/h)
      * Probability Value - 9.38704e-22

 * Probability #5: 
      * Class Label: Keep left
      * Probability Value - 3.22007e-23   

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


