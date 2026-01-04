Machine Learning Project:  
Iris-flower-classification

This program applies basic machine learning (classification) concepts on Fisher’s Iris Data to predict the species of a new sample of Iris flower.


Introduction


The dataset consists of 50 samples from each of three species of Iris  
  (Iris setosa, Iris virginica, and Iris versicolor).

- Four features were measured from each sample (in centimetres):
  - Length of the sepals  
  - Width of the sepals  
  - Length of the petals  
  - Width of the petals  


Working of the iris_decision_tree_classifier

- The program takes data from the `load_iris()` function available in the `sklearn` module.  
- The program then creates a decision tree based on the dataset for classification.  
- The user is then asked to enter the four parameters of the sample, and the prediction about the species of the flower is printed to the user.


Working of the iris_selfmade_KNN

- The program takes data from the `load_iris()` function available in the `sklearn` module.  
- The program then divides the dataset into training and testing samples in an 80:20 ratio randomly using the `train_test_split()` function available in the `sklearn` module.  
- The training sample space is used to train the program, and predictions are made on the testing sample space.  
- The accuracy score is then calculated by comparing with the correct results of the training dataset.
