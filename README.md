[//]: # (Image References)

[image1]: ./writeup_images/image_1.png "Original Image"
[image2]: ./writeup_images/image_2.png "Undistorted"
[image3]: ./writeup_images/image_3.jpg "Perspective Transformed Image"
[image4]: ./writeup_images/image_4.png "L Channel"
[image5]: ./writeup_images/image_5.png "S Channel"
[image6]: ./writeup_images/image_6.png "Combine Channel"
[image7]: ./writeup_images/image_7.png "Polynomial Fitted"
[image8]: ./examples/polynomial-drawn.png "Polynomial drawn"
[image9]: ./examples/highlighted-lane.png "Highlighted Lane"
[image10]: ./examples/combined-image.png "Combined Image"
[video1]: ./project_video_out.mp4 "Video"

# Vehicle Detection Project
---
### Histogram of Oriented Gradient (HOG)
---
#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I have a function in my code called `get_hog_features` which takes in the image as one of the parameters. I feed in random 1000 images in `non-vehicles` folder to extract the features for the non-vehicles and also the same for the `vehicles` folder to extract all the features for vehicles. The features are extracted in the function using `skimage.features` feature called `hog`. 

#### Sample Images and HOG Images
![alt text][image1]

---
#### 2. Explain how you settled on your final choice of HOG parameters

The way I settled with the final choices for the HOG parameters is through trial and errors. I ended up choosing colorspace `YCrCb` with `9` orientations using `2` cells per block with `ALL` HOG channels. I ended up getting `98.65%` test accuract on the tests. 

---
#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained the linear SVM with default parameters. The code I used to train the classifier is 
```
svc = LinearSVC()
svc.fit(X_train, y_train)
```
I got the accuracy of 98.65%

---
Sliding Window Search
---
#### 1. Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?

In the function `find_cars`, it takes parameters called `ystart` and `ystop`. `ystart` and `ystop` is the boundaries where the sliding windows does not go on top of `ystart` and also not below `ystop`. The way I decided to use the scale was through trial and errors. I found out that for smaller scale performs much better detecting the cars that are further awaya and the opposite for larger scale. Therefore, I ran many scales and combined the results later, piling the result on top of eachother. 

To get rid of the result for the false positives, i have a function called `add_heat`, where it keeps track of overlapping boxing. Later I use the return result from `add_heat` to set a threshold to get rid of false positives. 

#### Image with results from 3 different scaled windows
![alt text][image3]

#### Heatmap of the Image
![alt text][image4]

#### Heatmap after the threshold
![alt text][image5]

#### Assigning each label
Labels are assigned by using the function `scipy.ndimage.measurements.label()`
![alt text][image6]

#### Final detection area
![alt text][image7]

---
#### 2. Show some examples of test images to demonstrate how your pipeline is working. What did you do to optimize the performance of your classifier?

To help optimize the performance of my classifier, I split the `pix_per_cell` from `16` to `8`. I came to conclusion tha the accuracy gained from this was not significant enough to justify for the time it takes to run my classifier.

