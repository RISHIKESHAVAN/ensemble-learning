import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from collections import Counter

import emoji
from emot.emo_unicode import UNICODE_EMOJI,EMOTICONS_EMO,EMOJI_UNICODE


tweets = pd.read_csv("../data/cyberbullying_tweets.csv")


def extract_hashtags(series):
    regex = "#(\\w+)"
    hashtag_list = re.findall(regex, series)
    l=[]
    for x in hashtag_list:
        x = x.lower()
        x = re.sub(r"[^a-zA-Z ]+", '', x)
        l.append(x)
    if len(hashtag_list) > 0:
        return ' '.join(hashtag_list)
    else:
        return ''

def extract_tags(series):
    regex = "@(\\w+)"
    name_tag = re.findall(regex, series)
    l=[]
    for x in name_tag:
        x = x.lower()
        x = re.sub(r"[^a-zA-Z ]+", '', x)
        l.append(x)
    if len(l) > 0:
        return ' '.join(l)
    else:
        return ''

def extract_links(series):
    regex = r'(https?://\S+)'
    url_list = re.findall(regex, series)
    l=[]
    for x in url_list:
        x = x.lower()
        x = re.sub(r"[^a-zA-Z ]+", '', x)
        l.append(x)
    if len(url_list) > 0:
        return ' '.join(url_list)
    else:
        return ''

def extract_emojis(series):
    emoji_list = ''.join(c for c in series if c in emoji.UNICODE_EMOJI['en'])
    text_emoji =  ''.join(emoji.demojize(x) for x in emoji_list if x)
    regex = r"\:(.*?)\:"
    text_emoji =  re.findall(regex, text_emoji)
    if len(text_emoji) > 0:
        return ' '.join(text_emoji)
    else:
        return ''

def clean_text(text,dirt):
    dirt = dirt.split()
    for x in dirt:
        text = text.replace(x,'')
        text = " ".join(text.split())
    return text

def normalize_accent(string):
    string = string.replace('á', 'a')
    string = string.replace('à', 'a')
    string = string.replace('â', 'a')

    string = string.replace('é', 'e')
    string = string.replace('è', 'e')
    string = string.replace('ê', 'e')
    string = string.replace('ë', 'e')

    string = string.replace('î', 'i')
    string = string.replace('ï', 'i')

    string = string.replace('ö', 'o')
    string = string.replace('ô', 'o')
    string = string.replace('ò', 'o')
    string = string.replace('ó', 'o')

    string = string.replace('ù', 'u')
    string = string.replace('û', 'u')
    string = string.replace('ü', 'u')

    string = string.replace('ç', 'c')
    
    return string

def clean_tweet(text):
    # Remove Hashtag, Mention, https, www.asdfd, dsfadsf.com
    pattern = re.compile(r"(#[A-Za-z0-9]+|@[A-Za-z0-9]+|https?://\S+|www\.\S+|\S+\.[a-z]+|RT @)")
    text = pattern.sub('', str(text))
    text = " ".join(text.split())
    
    # Make all text lowercase
    text = text.lower()
    
    # Replace accented letters
    text = normalize_accent(text)

    # Lemmatize word and remove stopwords
    text = " ".join([lemma.lemmatize(word) for word in str(text).split() if word.isalpha() and word not in STOPWORDS])

    # Remove Punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    return text

def process_tweets(tweets):
	tweets['tweet_text'] = tweets['tweet_text'].str.lower()
	tweets['hash_tag']   = tweets['tweet_text'].apply(extract_hashtags)
	tweets['name_tag']   = tweets['tweet_text'].apply(extract_tags)
	tweets['url_tag']    = tweets['tweet_text'].apply(extract_links)
	tweets['emoji_tag']  = tweets['tweet_text'].apply(extract_emojis)

	# Remove the symbols and numbers from tweet text
	tweets['tweet_text'] = tweets['tweet_text'].apply(lambda x: re.sub(r"[^a-zA-Z ]+", '', x))

	# Remove the hashtags, mentions, url and emojis from tweet text
	tweets['tweet_text'] = tweets.apply(lambda row :clean_text(row['tweet_text'],row['name_tag']),axis=1)
	tweets['tweet_text'] = tweets.apply(lambda row :clean_text(row['tweet_text'],row['hash_tag']),axis=1)
	tweets['tweet_text'] = tweets.apply(lambda row :clean_text(row['tweet_text'],row['url_tag']),axis=1)

	# Clean the tweet
	tweets['clean_tweet'] = tweets['tweet_text']
	tweets['clean_tweet'] = tweets['clean_tweet'].apply(lambda text: clean_tweet(text))

	return tweets




STOPWORDS = set(stopwords.words('english'))
lemma = WordNetLemmatizer()
