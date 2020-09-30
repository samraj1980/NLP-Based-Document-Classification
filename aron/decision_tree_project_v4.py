# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 21:13:07 2019
@author: Sam
"""
import numpy as np
import pandas as pd
import re
import nltk
from sklearn.datasets import load_files
nltk.download('stopwords')
from nltk.corpus import stopwords
import pickle

# Reading the Csv file
df = pd.read_csv('C:/Users/aront/Downloads/crises_team134_training_data_positive_labels.csv', encoding = "ISO-8859-1")
print(df.columns)

# Extracting the Tweet_Text and Informativeness column
df1 = df[[' Tweet Text' , ' Informativeness']]

# Creating a new Column Crisis_ind to indicate if it is a crisis or Not based on Informativeness Column

# Function to default values for Crisis Ind
def ind_crisis(row):
        if row[' Informativeness'] == 'Not related' or row[' Informativeness'] ==  'Not applicable':
            return(0)
        if row[' Informativeness'] == 'Related - but not informative' or  row[' Informativeness'] ==  'Related and informative':
            return(1)

# Calling function to introduce the new column in dataframe
df1["Crisis_ind"] = df.apply(lambda row: ind_crisis(row), axis = 1)
print(df1.columns)

# Rename the column for reference later
df1.rename(columns = {' Tweet Text':'Tweet_Text'}, inplace = True)

# Check for any missing values in the columns
df1.isnull().any()

# Impute Missing Information in the columns
df1 = df1.fillna(method='ffill')


#Load the columns of data farme into X and y
y = df1.Crisis_ind
X = df1.Tweet_Text

# Creating "Bag of Words" by choosing the most prominent 1000 words and also set the minimum amount of times a word should have re occured.
# Also eliminates the stop words

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=1000, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(X).toarray()


# Applying the "TFIDF" method to resolve the issue caused by bag of words features
# Convert values obtained using the bag of words model into TFIDF values and assigns weightage to each word
# Method:
#TF = (Number of Occurrences of a word)/(Total words in the document)
#IDF(word) = Log((Total number of documents)/(Number of documents containing the word))

from sklearn.feature_extraction.text import TfidfTransformer
tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()

# Splitting into Test and Train Sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training Text classification using RandomForest Method
from sklearn.tree import DecisionTreeClassifier

#depth = int(len(X_train)/375)
#samples = int(len(X_train)/330)
depth = 10
samples = 10

classifier = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth = depth, min_samples_leaf = samples)
classifier.fit(X_train, y_train)

# Predicting using the classifier model
y_pred = classifier.predict(X_test)

# Evaluating the Performance of Model using Confusion matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))


############ Future Purposes ##################################################


# Saving the Trained Model as a pickle object in Python
with open('text_classifier', 'wb') as picklefile:
    pickle.dump(classifier,picklefile)

# Loading The module for Prediction Purposes
with open('text_classifier', 'rb') as training_model:
    model = pickle.load(training_model)

# Predicting the sentiment/classification using the Model
y_pred2 = model.predict(X_test)

print(confusion_matrix(y_test, y_pred2))
print(classification_report(y_test, y_pred2))
print(accuracy_score(y_test, y_pred2))
