# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 21:13:07 2019

@author: Sam
"""
import time
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer 
#from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Reading the Csv file 
df = pd.read_csv('C:/Users/Sam/Desktop/Georgia tech MA AI/Fall 2019/CSE 6242 - Data and Visual Analytics/Project/crises_team134_training_data_pos_neg_labels.csv')
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
 
#from sklearn.feature_extraction.text import CountVectorizer
#vectorizer = CountVectorizer(max_features=1000, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
#vectorizer = vectorizer.fit_transform(X).toarray()

#X = vectorizer

# TFIDF Vectorizer to vectorize the text 
stopset = set(stopwords.words('english'))
vectorizer_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000, use_idf=True, lowercase=True, strip_accents="ascii", stop_words=stopset).fit(X)

# Convert the df1.Tweet text from text to features 
X = vectorizer_ngram.transform(X)



# Applying the "TFIDF" method to resolve the issue caused by bag of words features
# Convert values obtained using the bag of words model into TFIDF values and assigns weightage to each word 

# Method:
#TF = (Number of Occurrences of a word)/(Total words in the document)
#IDF(word) = Log((Total number of documents)/(Number of documents containing the word))

#from sklearn.feature_extraction.text import TfidfTransformer
#tfidfconverter = TfidfTransformer()

#tfidconverter1 = tfidfconverter.fit_transform(X).toarray()

#X = tfidconverter1
# Convert the df1.Tweet text from text to features 
#X = vectorizer_ngram.transform(X)

print("y shape:", y.shape) 
print("x shape", X.shape)  


# Splitting into Test and Train Sets 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

start_time = time.time() #timer 


# Training Text classification using RandomForest Method with 500 trees

classifier = RandomForestClassifier(n_estimators=500, random_state=0)
classifier.fit(X_train, y_train) 

print("--- %s seconds ---" % (time.time() - start_time)) 

# Predicting using the classifier model 
y_pred = classifier.predict(X_test)

# Evaluating the Performance of Model using Confusion matrix 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

############ Future Purposes ##################################################

## Saving the Trained Model as a pickle object in Python 
#with open('text_classifier', 'wb') as picklefile:
#   pickle.dump(classifier,picklefile)
picklefile = open('random_classf', 'wb')
pickle.dump(vectorizer_ngram , picklefile)
pickle.dump(classifier, picklefile)    

#    