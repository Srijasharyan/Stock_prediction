
# Mask-No-Mask-Detection(PRML course project)

Predicts whether the person in an image is wearing a Mask or not


## Structure of the Project Folder
- main.py - Generates the dataset (data.csv) from the images.
- MinorProject.ipynb - The Project.
- Dataset - with_mask - images with mask without_mask
- data.csv - Dataset consisting of flatted nparray of 64 x 64 RGB images with target variable.

## Methodology

### Preprocessing and data description

#### Dataset
The given dataset contains various images of people with and without masks ,separated in a large
number of folders
![0026](https://user-images.githubusercontent.com/54506517/184806027-0bc03da8-534a-421e-ac3e-4be47416b4f1.jpg) <br/>
Person with mask

![0_0_caobingkun_0181](https://user-images.githubusercontent.com/54506517/184806171-334265a1-8fd5-4569-aa7b-a7df00e2a3ca.jpg) <br/>
Person without_mask

#### Data Preprocessing
We iterated through each folder and separated the masked and without mask dataset.Then we scaled the
images to 64 x 64 pixels size and converted the images to numpy arrays using asarray(img) and created a
dataframe and assigned labels as 0 and 1 for unmasked and masked data respectively.The datas were
shuffled using the sample() method.(This was done locally and submitted as main.py file

The above generated Data Frame was then used for model building and testing.
There are a total of 1264 data sets.
Then we have split the dataset into a training and a testing set using the train_test_split sklearn library by
taking 80% of the data in the training dataset and remaining 20 in the testing.

### Dimensionality reduction
Since the data contains a large number of independent variables we apply dimensionality
reduction technique of Principal Component analysis (PCA) , where the dimensions have been
reduced to 125 from 12288.
We could have used LDA but since the number of classes is only 2 and the features are
independent we have used PCA.
PCA is a machine learning method that is used for dimensionality reduction. The main idea of
principal component analysis (PCA) is to reduce the dimensionality of a data set consisting of
many variables correlated with each other, either heavily or lightly, while retaining the variation
present in the dataset, up to the maximum extent.PCA allows the collapsing of hundreds of spatial
dimensions into a handful of lower spatial dimensions while usually preserving 70% - 90% of the
important information .
LDA is like PCA which helps in dimensionality reduction, but it focuses on maximizing the
separability among known categories by creating a new linear axis and projecting the data points
on that axis.

### Using different Models
#### Support Vector Machine(SVM)
SVM works on smaller datasets, but on the complex ones, it
can be much stronger and powerful in building machine learning models.In the SVM algorithm, we plot
each data item as a point in n-dimensional space (where n is number of features you have) with the value
of each feature being the value of a particular coordinate. Then, we perform classification by finding the
hyper-plane that differentiates the two classes very well.Support Vectors are simply the coordinates of
individual observation. The SVM classifier is a frontier which best segregates the two classes
(hyper-plane/ line)
We first train the data using SVM classifier on non dimensional reduced dataset and we then calculate
the accuray.
- Accuracy by SVM without PCA :  0.9565217391304348
Now we train the model using reduced dataset obtained from PCA reduction and calculate the accuracy
- Accuracy by SVM with PCA : 0.9525691699604744

#### Multilayer Perceptron
A multilayer perceptron (MLP) is a class of feedforward artificial
neural network (ANN). An MLP consists of at least three layers of nodes: an input layer, a hidden layer
and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation
function. MLP utilizes a supervised learning technique called backpropagation for training.Its multiple
layers and non-linear activation distinguish MLP from a linear perceptron. It can distinguish data that is
not linearly separable
We first train the data using MLP classifier on non dimensional reduced dataset and we then calculate
the accuray.
- Accuracy by MLP without PCA : 0.5454545454545454
Now we train the model using reduced dataset obtained from PCA reduction and calculate the accuracy
- Accuracy by MLP with PCA : 0.8221343873517787


## Authors

- [@Puru-Raj](https://github.com/Puru-Raj)
- [@pratyaksh123](https://github.com/pratyaksh123)
