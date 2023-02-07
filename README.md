# Hums-and-Whistles-Audio-Classifier
Project to experiment with the machine learning pipeline. Classifies an audio file into one of two songs.

# 1 Problem formulation

The machine learning problem we are addressing is a supervised classification problem. We aim to predict if the song label of an audio file (hum or whistle of one of two songs) is Potter or Starwars through building a binary classifier. This is an interesting problem because individuals have different ways of humming or whistling a song, for instance, some may choose to hum as 'la-la-la' and others as 'hmm-hmm-hmm'. Moreover, the rhythm in which the song is hummed or whistled will differ due to differences in how well an individual knows the song as well as their musical timing. We aim to predict which of the two songs is being hummed or whistled despite these differences. 

# 2 Machine Learning pipeline

The final pipeline that is selected has the following structure:

input (audio data) --> transformation (feature extraction) --> transformation (feature selection) --> normalisation --> model --> output (prediction of song label)

The initial input to the pipeline is a collection of audio files of individuals humming or whistling to either Potter or Starwars. This data is then transformed by extracting nine features from the dimension-rich dataset. Feature selection is then used to select the most relevant features for the model. The features are then normalised. A model (that is trained then chosen through evaluating three families of models) takes as input the matrix of normalised features (X) and produces a prediction of the song label (y) based on what it has learnt from the training dataset.  


# 3 Transformation stage

There are two transformations which belong to the category of dimensionality reduction: feature extraction and feature selection.

For feature extraction, the input is raw audio files and the output is nine extracted features. These nine features inlcude some that I identified as being potentially relevant from the following paper:
Al-Maathidi, Muhammad M.. “Optimal feature selection and machine learning for high-level audio classification: a random forests approach.” (2017).

The thirteen extracted features are as follows: power, tempo, voice frequency, and the mean and standard deviation of pitch, spectral centroid, spectral bandwidth, onset and zero crossing rate. 

However, acknowledging that I do not possess domain expertise in audio processing, I used an additional transformation step, feature selection, to choose the features that were most relevant and to avoid using redudant predictors. For this we use recursive feature elimination (RFE) where a model is trained using different subsets of predictors and the subset of most relevant predictors are chosen. The chosen predictors will be the input for the model. For this transformation the input is the nine extracted features from the features extraction stage, the ouput is five selected features. 


# 4 Modelling
 

As we are adressing a classification problem, we will be using classification models to predict the song label of the audio file. Because we have access to the the labels of the audio files, we will be using 'supervised' learning models.
We train 3 different families of models: Support vector machine (SVM), k-nearest neighbour (kNN) and a Random Forest Classifier (RFC)

- SVM: creates a hyperplane that separates the data into classes. 
- kNN: a model where a sample is assigned to the same label as its k-nearest neighbours.
- RFC: an ensemble of decision tree models. A decision tree sequentially compares features against a threshold to eventually reach a classification. 

The model with the highest validation performance is chosen as the final model.


# 5 Methodology


To address our problem we will use a subset (70%) of the full dataset as the training dataset, the remaining 30% of the dataset is reserved for validation. The training dataset will be used to select features, train the models and choose hyper parameters. The validation dataset will be used after the models have been trained to compare their performance.

We will use the accuracy (proportion of correctly classified samples) as a measure of model performance. This is because for this ML problem, the cost of missclasifying Potter as Starwars is equal to the cost of missclassifying Starwars as Potter. In addition, the size of the two classes is almost equal, there is not class imbalance. Overall we aim to classify correctly as many samples as possible. The model which has the highest accuracy on the validation dataset will be chosen as the final model.

Note: we do not evaluate the future deployment performance of the model, a step known as 'testing' the model. This is because we are not creating a product. If we were to utilise the functionality provided by the model for a product, we would need collect data from the target population and see how the chosen model performs on this unseen data. 


# 6 Dataset

The raw data is a collection of audio files from QMUL MSc students on the 'Principles of Machine Learning' module. It is important to consider our target population. For the sake of this problem, which is to build a binary classifier and not a product that uses the functionality of the model, we will identify the target population as QMUL MSc students who take the aforementioned model. For this problem our dataset is representative of the target population. If our target population was broader, e.g., anyone of any age who wants to see if the song they are humming is Starwars or Potter, the data we have would not be representative. 

The raw audio files have high dimensionality and it is not practical to use this many predictors for our model. To build and validate our models we need to extract a smaller set of features, and obtain a feature matrix of size (n, m) where n is the number of items and m is the number of features. We also need to obtain a label vector which contains the song label for each item. 



# 7 Results and Conclusions


For validation of which model to choose, we used a set-approach. We chose this approach as oppose to k-fold validation because using k-fold would result in using the same data for validation that was used to select the features and used to choose the value of k in kNN i.e., did we select features or a value of k that were ideal for that dataset but might not be on unseen data? Therefore, the validation dataset was kept separate and not involved in feature selection or choosing the value of k for the kNN model. The performance of each model was then evaluated through its performance on the unseen validation dataset. 

The validation accuracy of the three models are close, and for the most part were higher with normalised data. The validation accuracy of the three models with normalised data are as follows:
RFC: 78.6%
kNN: 77.8%
SVM: 74.2%

The Random forest classifier and the kNN models both produced the highest validation accuracy of around 78%. The RFC model performs slightly better, therefore we choose RFC as our model. A validation accuracy of 78.6% means that the model is not a particularly strong classifier. If we consider that the worst binary classifier has an accuracy of 50%, the kNN model certainly performs better than this, but still makes the wrong prediction 21.4% of the time. 

A weakness that should be noted is that feature selection was carried out through ranking features using the ranking attribute `feature_importances` of the RFC model, which is the mean and standard deviation of accumulation of impurity decrease within each tree. These same features were then used for two different families of models. As the features were selected based off feature relevance on one model (or ensemble of models), this may introduce bias, i.e., the relevance of features may differ between different models. A high level of overfitting in the RFC model demonstrates how the feature selection was highly tuned to the training dataset for this particular model. Another limitation is that we did not optimise the number of features that should be used, the choice of 5 is somewhat arbritary. The limited dataset, where the model was only trained on 280 samples per class, likely contributed to poorer performance and lower validation accuracy. Additionally, proper processing of the audio data through utilising domain expertise would have likely lead to a better validation accuracy. 



