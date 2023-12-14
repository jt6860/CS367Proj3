# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 11:04:12 2023

@author: John Torres
"""
#PubNub imports
from pubnub.callbacks import SubscribeCallback
from pubnub.enums import PNStatusCategory, PNOperationType
from pubnub.pnconfiguration import PNConfiguration
from pubnub.pubnub import PubNub

#Sklearn imports
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

#Utility imports
import numpy as np
import pandas as pd
import time
from csv import DictWriter

#PubNub config
pnconfig = PNConfiguration()
pnconfig.subscribe_key = 'sub-c-d00e0d32-66ac-4628-aa65-a42a1f0c493b'
pnconfig.publish_key = 'pub-c-8429e172-5f05-45c0-a97f-2e3b451fe74f'
pnconfig.user_id = "my_custom_user_id"
pnconfig.connect_timeout = 5
pubnub = PubNub(pnconfig)

#primekeys is a collection of keys that are present in each tweet, used in subscribe callback message
primekeys = ['created_at', 'id', 'text', 'source', 'truncated', 'in_reply_to_screen_name', 'user', 'geo', 'coordinates', 'place', 'contributors', 'is_quote_status', 'quote_count', 'reply_count', 'retweet_count', 'favorite_count', 'entities', 'favorited', 'retweeted', 'possibly_sensitive', 'filter_level', 'lang', 'timestamp_ms']

class MySubscribeCallback(SubscribeCallback):

    def status(self, pubnub, status):
        pass
        # PubNub setup
        if status.operation == PNOperationType.PNSubscribeOperation \
                or status.operation == PNOperationType.PNUnsubscribeOperation:
            if status.category == PNStatusCategory.PNConnectedCategory:
                pass
                # Is no error or issue whatsoever
            elif status.category == PNStatusCategory.PNReconnectedCategory:
                pass
                # If subscribe temporarily fails but reconnects. This means
                # there was an error but there is no longer any issue
            elif status.category == PNStatusCategory.PNDisconnectedCategory:
                pass
                # No error in unsubscribing from everything
            elif status.category == PNStatusCategory.PNUnexpectedDisconnectCategory:
                pass
                # This is an error, retry will be called automatically
            elif status.category == PNStatusCategory.PNAccessDeniedCategory:
                pass
                # This means that Access Manager does not allow this client to subscribe to this
                # channel and channel group configuration. This is another explicit error
            else:
                pass
                # This is usually an issue with the internet connection, this is an error, handle appropriately
                # retry will be called automatically
        elif status.operation == PNOperationType.PNSubscribeOperation:
            # Heartbeat operations can in fact have errors, so it is important to check first for an error.
            # For more information on how to configure heartbeat notifications through the status
            # PNObjectEventListener callback, consult <link to the PNCONFIGURATION heartbeart config>
            if status.is_error():
                pass
                # There was an error with the heartbeat operation, handle here
            else:
                pass
                # Heartbeat operation was successful
        else:
            pass
            # Encountered unknown status type
 
    def presence(self, pubnub, presence):
        pass  # handle incoming presence data
    def message(self, pubnub, message):
        res = dict()
        for key, val in message.message.items():
            if key in primekeys:
                res[key] = val
        with open('output.csv', 'a', encoding="utf-8") as f_object:
            dictwriter_object = DictWriter(f_object, fieldnames=primekeys)
            dictwriter_object.writerow(res)
            f_object.close()
        pass  # handle incoming message data
        
subscribeCallback = MySubscribeCallback()
pubnub.add_listener(subscribeCallback)
pubnub.subscribe().channels('pubnub-twitter').execute()

#Increase time.sleep from (1) to higher number to increase amount of data gathered to .csv
time.sleep(100)
pubnub.unsubscribe_all()

#Read in .csv
csvdf = pd.read_csv('output.csv', names=primekeys)
print("Raw data:")
print(csvdf)
print("\nlength of data is", len(csvdf))
print("\nshape is", csvdf.shape)
dataset = csvdf[['text', 'lang']]

## Model Training
print("\nTraining Dataset:")
print(dataset)
X=dataset.text
Y=dataset.lang
# Separating the 95% data for training data and 5% for testing data
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.05, random_state =26105111)
vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectoriser.fit(X_train)
X_train = vectoriser.transform(X_train)
X_test  = vectoriser.transform(X_test)

def model_Evaluate(model):
    # Predict values for Test dataset
    y_pred = model.predict(X_test)
    # Print the evaluation metrics for the dataset.
    print(classification_report(y_test, y_pred, zero_division=np.nan))
    

BNBmodel = BernoulliNB()
BNBmodel.fit(X_train, y_train)
print("\nBernoulli Naive Bayes Classifier")
model_Evaluate(BNBmodel)
y_pred1 = BNBmodel.predict(X_test)

SVCmodel = LinearSVC()
SVCmodel.fit(X_train, y_train)
print("\nSupport Vector Machine")
model_Evaluate(SVCmodel)
y_pred2 = SVCmodel.predict(X_test)