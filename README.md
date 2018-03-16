# Spatial Pyramid Matching Scene Recognition
Trained a classifier to recognize 3000 images with 15 categories using Bag of Features model and Spatial Pyramid Matching algorithm. Improved accuracy from ~50% to ~70%.

# Install dependencies
All common python packages are needed (Numpy, Matplotlib,...). We also need OpenCV in this project.
```
pip install numpy matplotlib opencv-python
```
We will need to use SIFT built-int function of OpenCV to extract SIFT or SURF features from images, and since version OpenCV 3.x.x no longer includes these functions, one simple way is to downgrade to version 2.x.x. So first uninstall old version if already exists:
```
pip uninstall opencv-python
```
Then install this version: 
```
pip install opencv-contrib-python
```
Now we should be good to use SIFT/SURF descriptor in OpenCV.

Note that we can also use other features descriptor like HOG instead of SIFT.

# Overall
- Try classify on raw features (accuracy ~18% - 25%)
- Build a SIFT descriptor by constructing histogram of frequencies of "visual words". We find SIFT features for each image which is a 1D vector of size 128, then concatenate all these 1D vectors into a long 1D vector of the whole training set. Use K-means to cluster these data points from this vector into K groups. Now for each image, we have its SIFT features, assign these features into clusters that we've already clustered before, this will represent the histogram representation of frequencies of "visual words" for our image. Do this similarly for the rest of the images to form the features representation of our data before feeding into the classifier model. First, we will try K nearest neighbor. (accuracy ~50% - 60%)
- Build the SIFT histogram representation of "visual words" similarly as mentioned above, now we'll try to use multiclass Support Vector Machines as our classification model and compare the result. (accuracy ~60% - 70%)

Here is the intuition for constructing SIFT descriptor:
![screen shot 2018-03-16 at 12 21 34 am](https://user-images.githubusercontent.com/20756728/37503723-6675ccd0-28b0-11e8-8a93-87d517949f42.png)
![screen shot 2018-03-16 at 12 21 49 am](https://user-images.githubusercontent.com/20756728/37503730-6a12b664-28b0-11e8-86cc-db0d7b484ccc.png)

Finally, concatenate the 16 histograms together to get the final 128-element SIFT descriptor.
![screen shot 2018-03-16 at 12 22 01 am](https://user-images.githubusercontent.com/20756728/37503734-6cd6695e-28b0-11e8-8ffd-1d24020de9a1.png)

# Spatial Pyramid Matching
One drawback of Bag of Visual Words is, all local features are encoded into a single code vector ignoring the position of the feature descriptors, which means spatial information between words are discarded in the final code vector. Thus, to incorporate the spatial information into the final code vector, we can apply Spatial Pyramid Matching, a very simple but powerful idea proposed in [Lazebnik et al. 2006](http://www.di.ens.fr/willow/pdfs/cvpr06b.pdf).
