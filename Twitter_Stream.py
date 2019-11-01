import tweepy
import csv
import ssl
import time
from requests.exceptions import Timeout, ConnectionError
from requests.packages.urllib3.exceptions import *
import configparser

# Loading the config file at run time
config = configparser.ConfigParser()
config.read('config.ini')

# Add your Twitter API credentials
consumer_key = config.get('Twitter_Settings' , 'consumer_key')
consumer_secret = config.get('Twitter_Settings' , 'consumer_secret')
access_key =  config.get('Twitter_Settings' ,'access_key')
access_secret = config.get('Twitter_Settings' ,'access_secret')

print(consumer_key, consumer_secret, access_key, access_secret)

# Handling authentication with Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)

# Create a wrapper for the API provided by Twitter
api = tweepy.API(auth)

# Setting up the keywords, hashtag or mentions we want to listen
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
             "path hurricane"] 

# Set the name for CSV file  where the tweets will be saved
filename = "tweets_stream"


# We need to implement StreamListener to use Tweepy to listen to Twitter
class StreamListener(tweepy.StreamListener):

    def on_status(self, status):

        try:
            # saves the tweet object
            tweet_object = status

            # Checks if its a extended tweet (>140 characters)
            if 'extended_tweet' in tweet_object._json:
                tweet = tweet_object.extended_tweet['full_text']
            else:
                tweet = tweet_object.text

            '''Convert all named and numeric character references
            (e.g. &gt;, &#62;, &#x3e;) in the string s to the
            corresponding Unicode characters'''
            tweet = (tweet.replace('&amp;', '&').replace('&lt;', '<')
                     .replace('&gt;', '>').replace('&quot;', '"')
                     .replace('&#39;', "'").replace(';', " ")
                     .replace(r'\u', " "))

            # Save the keyword that matches the stream
            keyword_matches = []
            for word in keywords:
                if word.lower() in tweet.lower():
                    keyword_matches.extend([word])

            keywords_strings = ", ".join(str(x) for x in keyword_matches)

            # Save other information from the tweet
            user = status.author.screen_name
            timeTweet = status.created_at
            source = status.source
            tweetId = status.id
            tweetUrl = "https://twitter.com/statuses/" + str(tweetId)

            # Exclude retweets, too many mentions and too many hashtags
            if not any((('RT @' in tweet, 'RT' in tweet,
                       tweet.count('@') >= 2, tweet.count('#') >= 3))):

                # Saves the tweet information in a new row of the CSV file
                writer.writerow([tweet, keywords_strings, timeTweet,
                                user, source, tweetId, tweetUrl])

        except Exception as e:
            print('Encountered Exception:', e)
            pass


def work():

    # Opening a CSV file to save the gathered tweets
    with open(filename+".csv", 'w') as file:
        global writer
        writer = csv.writer(file)

        # Add a header row to the CSV
        writer.writerow(["Tweet", "Matched Keywords", "Date", "User",
                        "Source", "Tweet ID", "Tweet URL"])

        # Initializing the twitter streap Stream
        try:
            streamingAPI = tweepy.streaming.Stream(auth, StreamListener())
            streamingAPI.filter(track=keywords)

        # Stop temporarily when hitting Twitter rate Limit
        except tweepy.RateLimitError:
            print("RateLimitError...waiting ~15 minutes to continue")
            time.sleep(1001)
            streamingAPI = tweepy.streaming.Stream(auth, StreamListener())
            streamingAPI.filter(track=[keywords])

        # Stop temporarily when getting a timeout or connection error
        except (Timeout, ssl.SSLError,
                ConnectionError) as exception:
            print("Timeout/connection error...waiting ~15 minutes to continue")
            time.sleep(1001)
            streamingAPI = tweepy.streaming.Stream(auth, StreamListener())
            streamingAPI.filter(track=[keywords])

        # Stop temporarily when getting other errors
        except tweepy.TweepError as e:
            if 'Failed to send request:' in e.reason:
                print("Time out error caught.")
                time.sleep(1001)
                streamingAPI = tweepy.streaming.Stream(auth, StreamListener())
                streamingAPI.filter(track=[keywords])
            else:
                print("Other error with this user...passing")
                pass


if __name__ == '__main__':

    work()