
from cProfile import label
import re
import string
from tabnanny import verbose
from tokenize import Token
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from numpy import vectorize, zeros
import array
import numpy
import pandas as pd
from pandas import array
from pkg_resources import NullProvider
from sklearn.preprocessing import OneHotEncoder
# import tensorflow.python.compat
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Dense, Flatten
from keras.layers.embeddings import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import one_hot
# from keras_preprocessing.sequence import pad_sequences


# nltk.download()
# print(tensorflow.version)

#  Load Data

raw_data = pd.read_csv("Tweet Airlines reviews/Tweets.csv", header=0, delimiter=",")

# Discovery of Data

# print(raw_data)
# print(raw_data.shape)
# print(raw_data.info())
# print(raw_data.describe())


## Split data

# train_size = int(len(raw_data)* 0.5)
# # print(train_size)
# train_set = raw_data.head(train_size)
# test_size = int(len(raw_data)*0.5)
# # print(test_size)
# test_set = raw_data.tail(test_size)

# print(len(test_set), len(train_set))
##  Pre-process the data 

raw_tweet = raw_data["text"]
labels = raw_data["airline_sentiment"]
labels = numpy.array(labels)
# print(len(labels))
# print(labels[3])
# print(raw_tweet[1])

# print(raw_tweet[7])

def tweetCleaner(tweet):
    
    # Remove links via Regex
    clean_review = re.sub(r'http\S+', " ", str(tweet))
    clean_review = re.sub(r"[^A-Za-z]", " ", clean_review).split()

    # Remove the word VirginA;erica from review to avoid a high weight into the embedding model
    clean_review = [word for word in clean_review if not word in ["VirginAmerica", "AmericanAir"]]

    #  Remove stopwords
    stops = set(stopwords.words("english"))
    clean_review = [word for word in clean_review if not word in stops]

    #  Remove words lenght less than 2 
    stops = set(stopwords.words("english"))
    clean_review = [word for word in clean_review if not len(word) < 2]

    # print(clean_review)

    return clean_review

# res = tweetCleaner(raw_tweet[7])
clean_train = []
#  Clean Train Set
for i in raw_data["text"]:
    clean_train.append(tweetCleaner(i))

# clean_test = []
# #  Clean Test Set
# for i in test_set["text"]:
#     clean_test.append(tweetCleaner(i))

# print(clean_train)
# Create a doc of arrays containing tokens

# doc = []
# for tweet in raw_tweet:
#     doc.append(tweetCleaner(tweet))

# print(len(clean_train))
# print(len(clean_test))
# ## we tokeninze each sub docs and encode it : fit, encode , pad
# #smalll
doc_train = []
for ar in clean_train:
    for w in ar:
        doc_train.append(w)

# print(doc_train[:20])
# doc_test = []
# for ar in clean_test:
#     for w in ar:
#         doc_test.append(w)

# print(len(doc_test))
# print(len(doc_train))
# X_train, X_test, y_train, y_test = train_test_split(clean_train, clean_test, test_size=0.2, random_state=42)
x_train_size = int(len(clean_train)*0.8)
x_test_size = int(len(clean_train)*0.2)

X_train = clean_train[:x_train_size]
X_test = clean_train[x_train_size:]

y_test_size = int(len(labels)*0.2)
y_train_size = int(len(labels)*0.8)


y_train = labels[:y_train_size]
y_test = labels[y_train_size:]

# print(y_test_size)
# print(y_train_size)

# print(len(y_test))
# print(len(y_train))

# print(len(X_train), len(y_train))
# print(len(X_test), len(y_test))

# # print(len(temp))
tokens = Tokenizer()
tokens.fit_on_texts(doc_train)

maxLength = len(doc_train) + 1
print(type(maxLength))

encode_tweet = tokens.texts_to_sequences(doc_train)
vocab_size = len(tokens.word_index)+ 1
# print(tokens.word_index)
# print(len(encode_tweet))
# print(maxLength)
# print(vocab_size)
print("pad_sequences...")
padded = pad_sequences(encode_tweet, maxlen=int(maxLength), padding="post")

#  Shows the most used words (first 10)
# i = 0
# for word in tokens.word_counts:
#     if i <= 10:
#         print(word)
#     i += 1

#  Create CNN 1 Dimension Model

# model = tf.keras.Sequential()
# model.add(Embedding(vocab_size, 100, input_length=maxLength))
# model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=8, activation="relu"))
# model.add(tf.keras.layers.MaxPool1D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))


# # Compile network
# model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])
 

# model.fit(X_train, y_train, epochs=10, verbose= 2)

# #evaluate 
# loss, acc = model.evaluate(X_test, y_test, verbose=0)
# print(f'test accuracy: {acc*100}')

# print(model.summary())
