# Real Time Crisis Monitoring 



[![](https://img.shields.io/badge/authors-%40Sam%20Raj-blue)](https://www.linkedin.com/in/samraj-anand-jeyachandran-pmp-7b273a6/)



## Background

With the recent explosion of social media use, research has focused on extracting insights thereof. A lucrative endeavor is understanding and possibly predicting the future from this data, the benefits being material and moral. We will quickly detect crises where a population is suddenly at risk of mortal peril by analyzing Twitter data. To facilitate quick response, a real-time dashboard will monitor worldwide crises so aid organizations, journalists, supply chains, and others could mitigate harm, bodily or financial, and loss of life.
This project aims at building a software application to identify political instability or crisis where a population is in danger, briefly explain the crisis, and monitor it. The application will be fed tweets captured in near real-time. End-users will access it through an intuitive visualization to view ongoing crises worldwide. Currently, NGO’s and governments provide periodic reporting; it relies on manual efforts, is at risk of bias, and compromises on speed-to-market. In OECD countries, media outlets provide fast crisis reporting; however, non-OECD countries’ media cannot guarantee the same. Not having an accurate, up-to-date understanding of crises as they unfold is dangerous for those affected. They generally have narrower scope, whereas we seek to identify crisis more broadly. We seek to derive insights from real-time information, relying on Twitter instead of static reporting. Our success hinges on filling a different niche than historical reporting. We’ll use Tweets from all English-speaking sources to minimize bias and maximize firsthand data. Many organizations and individuals need to instantaneously know when crises occur and be able to respond immediately. Benefits include timely evacuations, quick police or military response, and minimal supply chain disruption. We will measure our software’s accuracy by comparing true positives to false positives for crises identified. We can also measure success after release by tracking repeat users, page views, and user satisfaction surveys. Unfortunately, it is not feasible to measure success through counting lives and dollars saved.


## Solution

This project involves harnessing tweets which are spatially and geographically linked to a crisis, classifying them using multiple algorithms, & adding situational context for each tweet and finally creating a visualization enabling users to monitor and predict developing crises across the Globe in a real time.

<table>
  <tr>
    <td>
      <img src="https://github.com/sgupta679/crisismonitor/blob/master/sam/Near_Final%20Code/Crisis%20Monitor%20Solution.jpg">
    </td>
  </tr>
</table>


### Training & Testing Data 
In order to train and test the Various classification Models, the data was downloaded into a csv file (crises_team134_training_data.csv) from the following link 

https://crisislex.org/data-collections.html

In essence, this project involved implementing the following components:

## 1. Components Description

### [A. Random_Forest_Classifier:](https://github.com/sgupta679/crisismonitor/blob/master/sam/Near_Final%20Code/Random_Forest_Classifier.py) 

The module involves reading the tweets from the aforementioned csv file into a Pandas data frame. It uses the "Tweet text" column as the data (X), leverages the "Informativeness" column and creates a column "Crisis_ind" to be used as the labels (Y). It then does tokenization (vectorize), normalization and feature extraction using the Natural language processing algorithms like "Bag of Words" & "TFIDF". As part of the process, the stop words were removed using the nltk library. The module extracts the 5000 most frequently used "pair of words" (enabled by ngram feature) across the tweets and creates a matrix of those features to be then used for text classification. It then splits the data into a training and test dataset with a ratio of 70:30.
The training dataset is used for fitting the random forest classifier models, and it uses about 500 trees in the process. Once the model is built, we use the testing dataset to evaluate the performance of the model using confusion matrix and arrive at the accuracy score. The trained model ('random_classf') and the vectorizer was pickled using the pickle library.
Sklearn library was used extensively in this module for feature extraction, model selection and Random Forest classification.

Sklearn library was used extensively in this module for feature extraction, model selection and Randomeforest classification.  
   
### [B. Naive_Bayes_Classifier:](https://github.com/sgupta679/crisismonitor/blob/master/sam/Near_Final%20Code/Naive_Bayes_Classifier.py) 
  This module involves similar steps like loading the csv file, tokenization (vectorize), normalization and feature extraction using the Natural language processing algorithms like "Bag of Words" & "TFIDF", etc and creates a matrix of 5000 most commonly used "pairs of words" as features to be used for building the model and splitting the dataset into training and testing datasets.
The training dataset is used for fitting the Naive Bayes MultinomialNB model for text classification. Once the model is built, we use the testing dataset to evaluate the performance of the model using AUROC method as the Naive Bayes algorithm only assigns a confidence % for each tweet that was classified as crisis. The trained model ('naive_classf') and the vectorizer was pickled using the pickle library.
Sklearn library was used extensively in this module as well for feature extraction, cross-validation, Naive Bayes classification and roc_auc metrics reporting.

### [C. SVM Classifier:](https://github.com/sgupta679/crisismonitor/blob/master/sam/Near_Final%20Code/SVM_Classifier.py)
All of the steps outlined above in the previous two models are performed here as well to extract the 5000 most commonly occurring pair of words as features to be used for training the model for text classification.
The training dataset is used for fitting the SVM model. Once the model is built, we use the testing dataset to evaluate the performance of the model using confusion matrix and arrive at the accuracy score. The trained model ('svm_classf') and the vectorizer was pickled using the pickle library.
Sklearn library was used extensively in this module as well for feature extraction, model selection, SVM classification. 

### [D. Real Time Twitter Ingestion Module:](https://github.com/sgupta679/crisismonitor/blob/master/sam/Near_Final%20Code/Twitter_Ingestion_Classification.py) 
Now that the classification models have been created, we proceed to building the Tweet Ingestion module. 

<table>
  <tr>
    <td>
      <img src="https://github.com/sgupta679/crisismonitor/blob/master/sam/Near_Final%20Code/Ingestion%20Module%20Flow.jpg">
    </td>
  </tr>
</table>


The twitter ingestion Module leverages tweepy library to listen to the twitter live streaming feed. A filter of 380 keywords related to crisis retreieved from https://crisislex.org/crisis-lexicon.html was used to filter the tweets during ingestion.  

 ## (i). Pre-processing Steps:
      1. Extract tweets that are only in English language.  
      2. Ignore Re-tweets
      3. Extract data from Extended tweet tag and also include hash tages. 
      4. Extract the crisis related keywords that was found in the tweet. 
      5. Extract Place, Country and type from tweets if they are present
      
 ## (ii). Text Classification & Sentiment Analysis      
   #### Vectorizing and Performing Real time classification: 
      1. Convert the Tweet words extracted into features using the vectorizer that was pickled. 
      2. Pass the extracted features and invoke the 3 trained models using the pickle library. The prediction from all the 3 models 
     (Random forest, Naive Bayes, SVM) are then determined.
   
   #### Apply Ensemble Algorithm: 
      1. Leverages the NLP library Spacy which loads the “en_core_web_sm” library for further processing. Using spacy the modules         extracts the situational context from the tweet focusing on entities (NORP, Facility, Org, GPE, LOC, EVENT, DATE, TIME) and numbers. 
 
      
   #### Apply Lemmatization Algorithm: 
     1. The model removes all URLs, @ references and Hashtags from the twitter message. 
     2. Leverages the NLP library Spacy which loads the “en_core_web_sm” library for further processing. Using spacy the modules         extracts the situational context from the tweet focusing on entities (NORP, Facility, Org, GPE, LOC, EVENT, DATE, TIME) and numbers. 
  
 ## (iii). Storing processed Tweets in Azure:
      1. After the tweets are processed, classified, lemmatized they are stored in an Azure database for downstream processing. 
The processed tweets are stored in Azure database on a real time basis. The Tableau Online visualization extracts the relevant information from the Azure SQL server and displays the crisis gradient information on a choropleth Map providing real time information on ongoing crisis and developing crisis for relevant stakeholders.

## E. Visualization Dashboard: 

<table>
  <tr>
    <td>
      <img src="https://github.com/sgupta679/crisismonitor/blob/master/sam/Near_Final%20Code/Visualization%20Dashboard.PNG">
    </td>
  </tr>
</table>

## 2. INSTALLATION & EXECUTION 
   ### CREATING THE APP IN TWITTER & GET TWITTER API SUBSCRIPTION 
    1. Ensure that the App is created in Twitter developer url 
    2. Create a subscription for twitter API and updated the credentials in the config.ini file. 
   ### RUNNING THE CLASSIFICATION MODELS  
    1. Ensure that the crises_team134_training_data.csv file is available in the folder. 
    2. Ensure that the following python packages are available before proceeding to next step 
       A. Pandas
       B. Numpy 
       C. nltk & download stopwords 
       D. sklearn
       E. pickle 
     3. Run the classification programs in the following order and ensure that the path to csv file is referenced correctly. Also ensure that the path to the pickle folder is referenced correctly. 
      A. Random_Forest_Classifier.py  
      B. Naive_Bayes_Classifier.py 
      C. SVM_Classifier.py  
   ### CREATING THE AZURE SQL SERVER DB 
      1. Make sure that the Azure SQL server database is created and the connection strings provided are updated into the config.ini              file.
      2. Ensure that the firewalls are opened for the IP address of the server/ machine where the Python programs are going to be run. 
   ### RUNNING THE TWITTER STREAMING MODULE 
      1. Ensure that the references to the Config.ini files and the 3 pickled classification models are correct. 
      2. Run the Twitter_Ingestion_Classification.py module. This module will harness the twitter information, perform the      classification and update the data into the Azure SQL DB. 
   ### VISUALIZATION
      1. You will find our Crisis Monitoring Dashboard at this link:
     
https://public.tableau.com/profile/savannah8260#!/vizhome/Project11-22b/CrisisMonitor2?publish=yes
