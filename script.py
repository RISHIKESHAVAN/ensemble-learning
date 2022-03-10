# import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import emoji
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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

from emot.emo_unicode import UNICODE_EMOJI,EMOTICONS_EMO,EMOJI_UNICODE

STOPWORDS = set(stopwords.words('english'))
lemma = WordNetLemmatizer()

'''
TEXT PRE-PROCESSING
'''
def extract_hashtags(series):
    '''
    Extraxt hashtag words in a string.
    '''
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
    '''
    Extract mentions (words starting with '@') in a string.
    '''
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
    '''
    Extract URLs in a string.
    '''
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
    '''
    Extract the emojis in a string and return the UNICODE format of them.
    '''
    emoji_list = ''.join(c for c in series if c in emoji.UNICODE_EMOJI['en'])
    text_emoji =  ''.join(emoji.demojize(x) for x in emoji_list if x)
    regex = r"\:(.*?)\:"
    text_emoji =  re.findall(regex, text_emoji)
    if len(text_emoji) > 0:
        return ' '.join(text_emoji)
    else:
        return ''

def clean_text(text,dirt):
    '''
    Cleans the `text` by replacing the characters passed in the `dirt` argument with an empty string.
    '''
    dirt = dirt.split()
    for x in dirt:
        text = text.replace(x,'')
        text = " ".join(text.split())
    return text

def normalize_accent(string):
    '''
    Replaces the accented characters with their canonical form.
    '''
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
    '''
    Used in the `process_tweets()`. Performs the following actions:
    - removes broken urls, hashtags, mentions 
    - converts text to lowercase
    - replaces accented characters using the `normalize_accent()`
    - removes stopwords from the text
    - lemmatizes the words
    - removes punctuations
    '''
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
    '''
    Used to process the tweet text.
    '''
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

'''
TF-IDF VECTORIZERS
'''
def hashtag_vectorizer(tweets):
    '''
    Returns the TF-IDF matrix of the hashtag words
    '''
    hshtg_vectorizer = TfidfVectorizer() 
    hshtg_vec = hshtg_vectorizer.fit_transform(tweets.hash_tag)
    tfidf_hashtags = hshtg_vectorizer.get_feature_names()
    hshtg_tfidf = pd.DataFrame(hshtg_vec.todense(), columns=tfidf_hashtags)
    return hshtg_tfidf


def emoji_vectorizer(tweets):
    '''
    Returns the TF-IDF matrix of the emojis
    '''
    emoji_vectorizer = TfidfVectorizer()
    emoji_vec = emoji_vectorizer.fit_transform(tweets.emoji_tag)
    tfidf_emojis = emoji_vectorizer.get_feature_names()
    emoji_tfidf = pd.DataFrame(emoji_vec.todense(), columns=tfidf_emojis)
    return emoji_tfidf

def tweet_vectorizer(tweets):
    '''
    Returns the TF-IDF matrix of the tweets text
    '''
    tfidf_vec = TfidfVectorizer(min_df=.0005, max_df=.90)
    tfidf = tfidf_vec.fit_transform(tweets.clean_tweet)
    tfidf_tweet_terms = tfidf_vec.get_feature_names()
    tweets_tfidf = pd.DataFrame(tfidf.todense(), columns=tfidf_tweet_terms)
    return tweets_tfidf

'''
FEATURE IMPORTANCE
'''

def decision_tree_feat_imp(X_train, y_train):
    '''
    Decision tree model used to get the important features
    '''
    grid_params = {'max_depth' : [int(x) for x in np.linspace(190, 220, num = 20)],#[85, 87, 89, 90, 91, 93],
                   'criterion' : ['entropy', 'gini'], 
                   'splitter' : ['best', 'random'] }
    dt_clf = DecisionTreeClassifier()
    dt_cv = GridSearchCV(estimator = dt_clf, 
                                   param_grid = grid_params, 
                                   cv = 7, 
                                   verbose=2, 
                                   n_jobs = -1, 
                                   scoring = 'f1_weighted')
    dt_cv.fit(X_train, y_train.values.ravel())
    dt_best = dt_cv.best_estimator_
    
    feat_imp = dt_best.feature_importances_
    return feat_imp

def compute_feat_imp(tweets):
    X = final_tweets.drop(['tweet_text','cyberbullying_type'], axis=1)
    Y = final_tweets.cyberbullying_type

    feat_imp = decision_tree_feat_imp(X,Y)

    feat_imp_index = feat_imp.argsort()[::-1]

    # we take only the first 1757 features since these are the ones with importance score > 0
    imp_cols = list(X.columns[feat_imp_index[:1757]])

    X_mod = X[imp_cols]

    df_for_models = pd.concat([X_mod, Y],axis=1)

    return df_for_models


'''
ENSEMBLE MODELS
'''

def decision_tree_classifier(X_train, X_test, y_train, y_test):
    '''
    DECISION TREE CLASSIFIER
    '''
    grid_params = {'max_depth' : [int(x) for x in np.linspace(180, 200, num = 20)],
                   'criterion' : ['entropy', 'gini'], 
                   'splitter' : ['best', 'random'] }
    
        
    dt_clf = DecisionTreeClassifier()
    dt_cv = GridSearchCV(estimator = dt_clf, 
                                   param_grid = grid_params, 
                                   cv = 7, 
                                   verbose=2, 
                                   n_jobs = -1, 
                                   scoring = 'f1_weighted')
    
    dt_cv.fit(X_train, y_train.values.ravel())
    
    dt_best = dt_cv.best_estimator_
    y_predicted = dt_best.predict(X_test)
    dt_accuracy = accuracy_score(y_test, y_predicted)
    dt_f1 = f1_score(y_test, y_predicted, average="weighted")
    
    return dt_accuracy, dt_f1


def random_forest_classifier(X_train, X_test, y_train, y_test):
    '''
    RANDOM FOREST CLASSIFIER
    '''
    grid_params = {'n_estimators': [50, 60, 70, 80],
                   'max_depth': [230, 240, 250, 260],
                   'min_samples_split': [20, 30, 40],
                   'min_samples_leaf': [1, 2, 3]}

    rf_clf = RandomForestClassifier()

    rf_cv = GridSearchCV(estimator = rf_clf, 
                         param_grid=grid_params, 
                         cv =5, 
                         verbose=1, 
                         scoring = 'f1_weighted', 
                         n_jobs=-1)
    
    rf_cv.fit(X_train, y_train.values.ravel())
    print("rf_cv.best_params_", rf_cv.best_params_)
    
    rf_best = rf_cv.best_estimator_
    y_predicted = rf_best.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_predicted)
    rf_f1 = f1_score(y_test, y_predicted, average="weighted")
    print("rf_accuracy, rf_f1", rf_accuracy, rf_f1)

    return rf_accuracy, rf_f1

def xg_boost_classifier(X_train, X_test, y_train, y_test):
    '''
    XG BOOST CLASSIFIER
    '''
    grid_params = {"n_estimators" : [int(x) for x in np.linspace(50, 500, num = 20)],
        "learning_rate" : [ 0.15, 0.20, 0.25, 0.30 ] ,
        "max_depth" : [int(x) for x in np.linspace(100, 250, num = 20)],
        "min_child_weight" : [ 1, 3, 5, 7 ],
        "gamma" : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
        "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] }

    xgb_clf = XGBClassifier()

    xgb_cv = RandomizedSearchCV(estimator = xgb_clf, 
                                    param_distributions = grid_params,  
                                    n_iter=50,
                                    cv=2,
                                    random_state=42, 
                                    n_jobs = -1, 
                                    scoring = 'f1_weighted')
    xgb_cv.fit(X_train, y_train.values.ravel())
    
    xgb_best = xgb_cv.best_estimator_

    y_predicted = xgb_best.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, y_predicted)
    xgb_f1 = f1_score(y_test, y_predicted, average="weighted")

    return xgb_accuracy, xgb_f1

def run(tweets):
    '''
    Main function that should be called with the tweets dataframe.
    '''
    # processes the tweets in the dataframe
    tweets = process_tweets(tweets)

    # creates the tf-idf matrix for the extracted hashtags
    hshtg_tfidf = hashtag_vectorizer(tweets)

    # creates the tf-idf matrix for the extracted emojis
    emoji_tfidf = emoji_vectorizer(tweets)

    # creates the tf-idf matrix for the tweets
    tweets_tfidf = tweet_vectorizer(tweets)

    # combine the different tf-idf matrices into one
    final_tweets = pd.concat([tweets[['tweet_text','cyberbullying_type']], tweets_tfidf, hshtg_tfidf, emoji_tfidf], axis=1)

    # drop the duplicate columns
    final_tweets = final_tweets.loc[:,~final_tweets.columns.duplicated()]

    final_tweets = compute_feat_imp(final_tweets)

    # multi-classification dataset
    X = final_tweets.drop(['cyberbullying_type'], axis=1)
    Y = final_tweets.cyberbullying_type
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # binary classification dataset
    binary_df = final_tweets.copy(deep=True)
    binary_df["cyberbullying_type"] = binary_df["cyberbullying_type"].apply(lambda x: "not_cyberbullying" if x == "not_cyberbullying" else "cyberbullying")
    X_bin = binary_df.drop(['cyberbullying_type'], axis=1)
    Y_bin = binary_df.cyberbullying_type
    X_bin_train, X_bin_test, y_bin_train, y_bin_test = train_test_split(X_bin, Y_bin, test_size=0.3, random_state=42)

    # hyper-parameter tuning
    # decision tree
    dt_a_m, dt_f1_m = decision_tree_classifier(X_train, X_test, y_train, y_test)
    dt_a_b, dt_f1_b = decision_tree_classifier(X_bin_train, X_bin_test, y_bin_train, y_bin_test)

    # random forest
    rf_a_m, rf_f1_m = random_forest_classifier(X_train, X_test, y_train, y_test)
    rf_a_b, rf_f1_b = random_forest_classifier(X_bin_train, X_bin_test, y_bin_train, y_bin_test)

    # xg boost
    rf_a_m, rf_f1_m = xg_boost_classifier(X_train, X_test, y_train, y_test)
    rf_a_b, rf_f1_b = xg_boost_classifier(X_bin_train, X_bin_test, y_bin_train, y_bin_test)

# import the dataset
tweets = pd.read_csv("../data/cyberbullying_tweets.csv")
run(tweets)







