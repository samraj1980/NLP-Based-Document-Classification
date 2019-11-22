"""
Spyder Editor

This is a temporary script file.

"""
import time
#import numpy as np
import pandas as pd
import nltk 
nltk.download('stopwords')  #First time after you have installed nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.cross_validation import train_test_split
from sklearn import naive_bayes 
from sklearn.metrics import roc_auc_score
import pickle

# Reading the training Csv file 
df = pd.read_csv('C:/Users/Sam/Desktop/Georgia tech MA AI/Fall 2019/CSE 6242 - Data and Visual Analytics/Project/crises_team134_training_data_pos_neg_labels.csv')
print(df.columns)

# Extracting the Tweet_Text and Informativeness column
df1 = df[[' Tweet Text' , ' Informativeness']]

# Creating a new Column Cris_ind to indicate if it is a crisis or Not based on Informativeness Column

# Function to default values for Crisis Ind
def ind_crisis(row): 
        if row[' Informativeness'] == 'Not related' or row[' Informativeness'] ==  'Not applicable':
            return(0)
        if row[' Informativeness'] == 'Related - but not informative' or  row[' Informativeness'] ==  'Related and informative':
            return(1)
               
df1["Crisis_ind"] = df.apply(lambda row: ind_crisis(row), axis = 1) 
print(df1.columns)

# Rename the column for reference later 
df1.rename(columns = {' Tweet Text':'Tweet_Text'}, inplace = True)

# Check for any missing values in the columns 
df1.isnull().any()

# Impute Missing Information in the columns 
df1 = df1.fillna(method='ffill')

# Tweet information is the predictor X & Crisis_ind is the response/dependent variable y

X = df1["Tweet_Text"]

# TFIDF Vectorizer to vectorize the text 
stopset = set(stopwords.words('english'))
vectorizer_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000, use_idf=True, lowercase=True, strip_accents="ascii", stop_words=stopset).fit(X)

# The dependent variable will be crisis_ind 0 (Non crisis Related) and 1 (Crisis Related) 
y = df1.Crisis_ind

# Convert the df1.Tweet text from text to features 
X = vectorizer_ngram.transform(X)

print("y shape:", y.shape) 
print("x shape", X.shape)  

# Test train Split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

start_time = time.time() #timer 

# Training a Naive Bayes Classifier 
naive_classf = naive_bayes.MultinomialNB()
naive_classf.fit(X_train, y_train) 

print("--- %s seconds ---" % (time.time() - start_time)) 

# Testing Model's accuracy using AUROC
print("accuracy using roc_auc:", roc_auc_score(y_test, naive_classf.predict_proba(X_test)[:,1]) * 100,"%")

threshold = roc_auc_score(y_test, naive_classf.predict_proba(X_test)[:,1])

y_pred = naive_classf.predict_proba(X_test)[:,1]


# Setting a threshold arrived by AUROC to make the TPR as 100%  (No False Positives)
for i in range(len(y_pred)):
    if round(y_pred[i]) < threshold:
        y_pred[i] = 0
    else: 
        y_pred[i] = 1

# Evaluating the Performance of Model using Confusion matrix 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))


############ Future Purposes ##################################################


# Saving the Trained Model as a pickle object in Python 
picklefile = open('naive_classf', 'wb')
pickle.dump(vectorizer_ngram , picklefile)
pickle.dump(naive_classf,picklefile)


