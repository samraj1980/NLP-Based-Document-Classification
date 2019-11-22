import tweepy
import json
from unidecode import unidecode
import time
import configparser
import pickle
import pyodbc 
from scipy.stats import stats
import spacy
from tqdm import *
import re


# Loading the config file at run time
config = configparser.ConfigParser()
config.read('config.ini')

# Add your Twitter API credentials
consumer_key = config.get('Twitter_Settings' , 'consumer_key')
consumer_secret = config.get('Twitter_Settings' , 'consumer_secret')
access_key =  config.get('Twitter_Settings' ,'access_key')
access_secret = config.get('Twitter_Settings' ,'access_secret')

# MSAZURE SQlSERVER Connection 
Driver='{SQL Server}'
server=config.get('Sql_server_Settings' , 'server')
database=config.get('Sql_server_Settings' , 'database')
username=config.get('Sql_server_Settings' , 'username')
password=config.get('Sql_server_Settings' , 'password')
 
conn = pyodbc.connect('DRIVER='+Driver+';SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
cursor = conn.cursor()

conn1 = pyodbc.connect('DRIVER='+Driver+';SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
cursor1 = conn1.cursor()

# Loading the Naive_Bayes Classifier_ Model for prediction   
file = open('naive_classf', 'rb')
vectorizer_ngram = pickle.load(file)
model = pickle.load(file)

# Loading the Random Forest Classifier_ Model for prediction   
file1 = open('random_classf', 'rb')
vectorizer_ngram1 = pickle.load(file1)
model1 = pickle.load(file1)

# Loading the SVM Classifier_ Model for prediction   
file2 = open('svm_classf', 'rb')
vectorizer_ngram2 = pickle.load(file2)
model2 = pickle.load(file2)




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
             "path hurricane", "Breaking", "shooting"] 


class updation():

    def on_data(row):

############# PERFORM BOW & TFIDF . APPLY CLASSIFICATION MODELS FOR PREDICTION   ###########################################
        tweet_text = row
        tweet_text = tweet_text.replace("(", "")
        tweet_text = tweet_text.replace(")", "")

        try:
             # Save the keyword that matches the stream
            keyword_matches = []
            for word in keywords:
                if word.lower() in tweet_text.lower():
                    keyword_matches.extend([word])

            keywords_strings = ", ".join(str(x) for x in keyword_matches)
            
            if len(keyword_matches) >= 1: 
                
                # Convert the tweet text from text to features before applying naivebayes classification 
                tweet2 = [tweet_text]
                tweet1 = vectorizer_ngram.transform(tweet2)
                Naive_Bayes_Ind = model.predict_proba(tweet1)[:,1]
                Naive_Bayes_Ind = Naive_Bayes_Ind[0]
                
                # Convert the tweet text from text to features before applying Random Forest classification 
                tweet4 = [tweet_text]
                tweet3 = vectorizer_ngram1.transform(tweet4)
                Random_Forest_Ind = model1.predict(tweet3)[0]
                Random_Forest_Ind = int(Random_Forest_Ind)
                
                # Convert the tweet text from text to features before applying Random Forest classification 
                tweet6 = [tweet_text]
                tweet5 = vectorizer_ngram2.transform(tweet6)
                SVM_Ind = model2.predict(tweet5)[0]
                SVM_Ind = int(SVM_Ind)
                
#######################     ENSEMBLE PREDICTION FROM MODELS  ####################################################################                 # Ensemble Algorithm 
                
                # Seting a threshold for Naive_Bayes_Ind
                if Naive_Bayes_Ind >= 0.96:
                    Naive_Ind = 1
                else: 
                    Naive_Ind = 0     
                
#               print(Naive_Ind, Random_Forest_Ind, SVM_Ind)
                ensemble_Ind = stats.mode([Naive_Ind, Random_Forest_Ind, SVM_Ind], nan_policy='propagate').mode
                ensemble_Ind = int(ensemble_Ind[0])
#                print(ensemble_Ind)


#######################     LEMMATIZATION     ########################################################################## 

                 # removing URLS
                tweet_text1 = re.sub(u'http\S+', u'', tweet_text) 
                
                # removing @... 
                tweet_text1 = re.sub(u'(\s)@\w+', u'', tweet_text1 )
                tweet_text1 = re.sub(u'@\w+', u'', tweet_text1 )
                
                # removing hashtags
                tweet_text1 = re.sub(u'#', u'',tweet_text1 )    
                
                #Tokenizing with SpaCy 
                nlp = spacy.load('en_core_web_sm')
                spacy_tweets = nlp(tweet_text1.lower())
                
                # Keep Only Content words - focus on entities and numbers 
                useful_entities = [u'NORP', u'FACILITY', u'ORG', u'GPE', u'LOC', u'EVENT', u'DATE', u'TIME']
                
                content_tweets = [] 
                for token in spacy_tweets: 
                     if ((token.ent_type_ in useful_entities)  
                        or (token.pos_ == u'NUM') 
                        or (token.lower_ in keywords)):
                        content_tweets.append(token)
                
                content_tweet = ""
                
                for content in content_tweets:
                    content_tweet = content_tweet  + str(content) + ";" 
                
                #Insert into MS Azure 
                cursor1.execute("UPDATE tweets SET keywords_strings = ? , Naive_Bayes_Ind = ?, Random_Forest_Ind = ? , SVM_Ind = ? , ensemble_Ind = ? , Context_Entities = ? where tweet_text = ?", 
                (keywords_strings, Naive_Bayes_Ind, Random_Forest_Ind, SVM_Ind, ensemble_Ind, content_tweet, tweet_text ))
                                  
                #Commit DB transactions
                conn1.commit()
    
        except KeyError as e:
            print(str(e))
        return(True)

    def on_error(status):
        print(status)


while True: 
    
    try:
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_key, access_secret)
        cursor.execute("select tweet_text from tweets where ensemble_Ind is null order by created_at")
        
        for row in cursor:
#            print(row[0])
            status = updation.on_data(row[0])

        
    except Exception as e:
        print(str(e))
        time.sleep(5)
