import tweepy
import json
from unidecode import unidecode
import time
import configparser
import pickle
import pyodbc 
from scipy.sparse.csr import csr_matrix
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


import pandas as pd
import numpy as np 
import spacy
#from spacy.tokens.doc import Doc
#import inspect
from sklearn.feature_extraction.text import TfidfVectorizer 
#from textacy.vsm import Vectorizer
#import textacy.vsm
#import scipy.sparse as sp
from tqdm import *
import re

tweets = pd.read_csv('C:/users/Sam/Desktop/Georgia tech MA AI/Fall 2019/CSE 6242 - Data and Visual Analytics/Project/crises_team134_training_data_pos_neg_labels.csv', encoding = 'ISO-8859-1')

keywords = ["flood crisis","victims","flood victims","flood powerful","powerful storms","hoisted flood","storms amazing","explosion","amazing rescue",
            "rescue women","flood cost","counts flood","toll rises","braces river","river peaks","crisis deepens","prayers","thoughts prayers",                 
            "affected tornado","affected","death toll","tornado relief","photos flood","water rises","toll","flood waters","flood appeal",
            "victims explosion","bombing suspect","massive explosion","affected areas","praying victims","injured","please join","join praying",
            "prayers people","redcross","text redcross","visiting flood","lurches fire","video explosion","deepens death","opposed flood","help flood",
             "died explosions","marathon explosions","flood relief","donate","first responders","flood affected","donate cross","braces",
             "tornado victims","deadly","prayers affected","explosions running","evacuated","relief","flood death","deaths confirmed",
             "affected flooding","people killed","dozens","footage","survivor finds","worsens eastern","flood worsens","flood damage","people dead",
             "girl died","flood","donation help","major flood","rubble","another explosion","confirmed dead","rescue","send prayers",
             "flood warnings","tornado survivor","damage","devastating","flood toll","affected hurricane","prayers families","releases photos",
             "hundreds injured","inundated","crisis","text donation","redcross give","recede","bombing","massive","bombing victims",
             "explosion ripped","gets donated","donated victims","relief efforts","news flood","flood emergency","give online","fire flood",
             "huge explosion","bushfire","torrential rains","residents","breaking news","redcross donate","affected explosion","disaster",
             "someone captured","tragedy","enforcement","people injured","twister","blast","crisis deepens","injuries reported","fatalities",
             "donated million","donations assist","dead explosion","survivor","death","suspect dead","peaks deaths","love prayers",
             "explosion fertiliser","explosion reported","return home","evacuees","large explosion","firefighters","morning flood","praying",
             "public safety","txting redcross","destroyed","displaced","fertilizer explosion","unknown number","donate tornado","retweet donate",
             "flood tornado","casualties","climate change","financial donations","stay strong","dead hundreds","major explosion","bodies recovered",
             "waters recede","response disasters","victims donate","unaccounted","fire fighters","explosion victims","prayers city",
             "accepting financial","torrential","bomber","disasters txting","explosion registered","missing flood","volunteers","brought hurricane",
             "relief fund","help tornado","explosion fire","ravaged","prayers tonight","tragic","enforcement official","saddened",
             "dealing hurricane","impacted","flood recovery","stream","dead torrential","flood years","nursing","recover","responders","massive tornado",
             "buried alive","alive rubble","crisis rises","flood peak","homes inundated","flood ravaged","explosion video","killed injured","killed people",
             "people died","missing explosion","make donation","floods kill","tornado damage","entire crowd","cross tornado","terrifying",
             "need terrifying","even scary","cost deaths","facing flood","deadly explosion","dead missing","floods force","flood disaster",
             "tornado disaster","medical examiner","help victims","hundreds homes","severe flooding","shocking video","bombing witnesses","magnitude",
             "firefighters police","fire explosion","storm","flood hits","floodwaters","emergency","flash flood","flood alerts","crisis unfolds",
             "daring rescue","tragic events","medical office","deadly tornado","people trapped","police officer","explosion voted","lives hurricane",
             "bombings reports","breaking suspect","bombing investigation","praying affected","reels surging","surging floods","teenager floods",
             "rescue teenager","appeal launched","explosion injured","injured explosion","responders killed","explosion caught","city tornado",
             "help text","name hurricane","damaged hurricane","breaking arrest","suspect bombing","massive manhunt","releases images","shot killed",
             "rains severely","house flood","live coverage","devastating tornado","lost lives","reportedly dead","following explosion","remember lives",
             "tornado flood","want help","seconds bombing","reported dead","imminent","rebuild","safe hurricane","surviving","injuries","prayers victims",
             "police suspect","warning","help affected","kills forces","dead floods","flood threat","military","flood situation","thousands homes",
             "risk running","dead injured","dying hurricane","loss life","thoughts victims","bombing shot","breaking enforcement","police people",
             "video capturing","feared dead","terrible explosion","prayers involved","reported injured","seismic","victims waters","flood homeowners",
             "flood claims","homeowners reconnect","reconnect power","power supplies","rescuers help","free hotline","hotline help","please stay",
             "investigation","saddened loss","identified suspect","bombings saddened","killed police","dead","praying community","registered magnitude",
             "leave town","reported explosion","heart praying","life heart","prepare hurricane","landfall","crisis worsens","arrest","bombing case",
             "suspect run","communities damaged","destruction","levy","tornado","hurricane coming","toxins flood","release toxins","toxins",
             "supplies waters","crisis found","braces major","government negligent","attack","hurricane","rebuilt communities","help rebuilt","rebuilt",
             "rescuers","buried","heart prayers","flood levy","watch hurricane","victims lost","soldier","waiting hurricane","run massive","high river",
             "terror","memorial service","terror attack","coast hurricane","terrified hurricane","aftermath","suspect killed","suspect pinned",
             "lost legs","hurricane category","names terrified","authorities","assist people","hurricane black","unknown soldier","events","safety",
             "troops","disaster relief","cleanup","troops lend","effected hurricane","time hurricane","saying hurricane","praying families","dramatic",
             "path hurricane", "Breaking"] 

tweets.head() 


# A. Preprocessing
tweets = tweets.dropna()

# removing URLS
tweets.Tweet_Text = tweets.Tweet_Text.apply(lambda x: re.sub(u'http\S+', u'', x))

# removing @... 
tweets.Tweet_Text = tweets.Tweet_Text.apply(lambda x: re.sub(u'(\s)@\w+', u'', x))
tweets.Tweet_Text = tweets.Tweet_Text.apply(lambda x: re.sub(u'@\w+', u'', x))

# remove u'RT'
tweets.Tweet_Text = tweets.Tweet_Text.replace(u'RT:', u'')

#Tokenizing with SpaCy 
nlp = spacy.load('en')
spacy_tweets = []

for doc in nlp.pipe(tweets.Tweet_Text, n_threads = -1):
    spacy_tweets.append(doc)

# Keep Only Content words - focus on entities and numbers 
useful_entities = [u'NORP', u'FACILITY', u'ORG', u'GPE', u'LOC', u'EVENT', u'DATE', u'TIME']

content_tweets = [] 
for single_tweet in tqdm(spacy_tweets):
    single_tweet_content = []
    for token in single_tweet: 
        if ((token.ent_type_ in useful_entities)  
            or (token.pos_ == u'NUM') 
            or (token.lower_ in keywords)):
            single_tweet_content.append(token)
    content_tweets.append(single_tweet_content)

# B. Getting the TFIDF score for Content words 

stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', use_idf=True, lowercase=True, strip_accents="ascii", stop_words=stopset)
term_matrix = vectorizer.fit_transform(tweets.Tweet_Text)

# Get all the feature_names 
feature_names = vectorizer.get_feature_names() 

# Create a dictionary between Feature_names and Index 
feature_name = {} 
for i , feature in enumerate(feature_names):
    feature_name[feature] = i

# Create a dictionary between Feature_name and its tfidf score  
    
tfidf_dict = {} # Features and the TFIDF score 
content_vocab = [] # List of all features 
for tweet in content_tweets: 
    for token in tweet: 
        if token.lemma_.lower() not in tfidf_dict: 
            if token.lemma_.lower() in feature_name.keys():
                content_vocab.append(token.lemma_.lower())
                tfidf_dict[token.lemma_.lower()] = np.max(term_matrix[:,feature_name[token.lemma_.lower()]])

# Test whether the dictionary is built correctly 
#for key in sorted(tfidf_dict)[500:505]:
#    print ("WORD:" + str(key) + " -- tf-idf SCORE:" +  str(tfidf_dict[key]))    
  
    
    
 # C. Content Word Based Tweet Summarization (COWTS) 
from pymprog import *
begin('COWTS')   
x = var('x', len(spacy_tweets), bool)
y = var('y', len(content_vocab), bool)
    
# Defining the equation that needs to be maximized  
maximize(sum(x) + sum([tfidf_dict[content_vocab[j]]*y[j] for j in range(len(y))]));
# 
# Was 150 for the tweet summary, 
# But generated a 1000 word summary for CONABS


# First Constraint 
L = 1000
sum([x[i]*len(spacy_tweets[i]) for i in range(len(x))]) <= L; 
    

def content_words(i):
    '''Given a tweet index i (for x[i]), this method will return the indices of the words in the 
    content_vocab[] array
    Note: these indices are the same as for the y variable
    '''
    tweet = spacy_tweets[i]
    content_indices = []
    
    for token in tweet:
        if token.lemma_.lower() in content_vocab:
            content_indices.append(content_vocab.index(token.lemma_.lower()))
    return content_indices    

def tweets_with_content_words(j):
    '''Given the index j of some content word (for content_vocab[j] or y[j])
    this method will return the indices of all tweets which contain this content word
    '''
    content_word = content_vocab[j]
    
    index_in_term_matrix = feature_name[content_word]
    
    matrix_column = term_matrix[:, index_in_term_matrix]
    
    return np.nonzero(matrix_column)[0]

# Second constaint 
for j in range(len(y)):
    sum([x[i] for i in tweets_with_content_words(j)])>= y[j]    

# Third constraint 
for i in range(len(x)):
    sum(y[j] for j in content_words(i)) >= len(content_words(i))*x[i]

solve()


result_x =  [value.primal for value in x]
result_y = [value.primal for value in y]    

end()

len(chosen_tweets[0]), len(chosen_words[0])


# Pritning the tweets that have been choosen 
for i in chosen_tweets[0]:
    print ('--------------')
    print(spacy_tweets[i])
