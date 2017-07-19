[//]: # (Image References)

[image1]: ./writeup-images/image_1.png "Image 1"
[image2]: ./writeup-images/image_2.png "Image 2"
[image3]: ./writeup-images/image_3.png "Image 3"
[image4]: ./writeup-images/image_4.png "Image 4"
[image5]: ./writeup-images/image_5.png "Image 5"
[image6]: ./writeup-images/image_6.png "Image 6"
[image7]: ./writeup-images/image_7.png "Image 7"
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
### Sliding Window Search
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

---
### Video Implementation
---
#### 1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

The video file is in this repository. The name of the file is `project_video_out.mp4`

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The way I combined the overlapping boxes are found in my function called `add_heat`. 
```
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap
```
I iterated the `add_heat` function by importing `deque` from `collections` to push array of rectangles for the last 8 frames. Than I added the heat of the last eight frames. This significantly reduced false positives and also reduced the jitter of the rectangles in the video.
```
def add_heat_deque(heatmap, rectangles_deque):
    # Iterate through list of bboxes
    for bbox_list in rectangles_deque:
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    # Return updated heatmap
    return heatmap
```
```
rectangles_deque = deque(maxlen=8)
```
```
rectangles_deque.append(rectangles)
```

And once we finish adding the heatmap, the way lessened the false positive is by adding the threshold to heatmap. The way that is done is by my function called `apply_threshold`.
```
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap
``` 

---
### Discussion
---
#### 1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

I found out the image pipeline fails in the video when the car is going over a large bump. The reason it fails is because when it goes over the bump, the video jumps and messes up the masking we have for the sliding window. We can fix that by making the sliding windows `ystart` and `yend` larger, but thought the trade off was not worth the time it takes to compute extra windows.

Also our classifier has an accuracy of 98.65% which looks good but actaully it returns about 13 false positives in a frame. We have to hope that the thresholding of the heatmap catches all the false positives. 