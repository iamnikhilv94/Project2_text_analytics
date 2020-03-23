import pandas as pd
import numpy as np
import os
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

os.getcwd()
os.chdir('/Users/nikhilviswanath/Documents/Python_data')

data=pd.read_csv('text_analytics.csv')

data.isnull().values.any()
data.isnull().sum()

data.dropna(axis=0,how='any',inplace=True)
data.shape
data.dtypes

text="""Hello Mr. Smith, how are you doing today? The weather is great, and city is awesome.The sky is pinkish-blue. You shouldn't eat cardboard"""
tokenized_text=sent_tokenize(text)
print(tokenized_text)

tokenized_word=word_tokenize(text)
tokenized_word[10]

tokenized_word[10]='Rahul'
distribution=FreqDist(tokenized_word)
type(distribution)
distribution.most_common(3)

distribution.plot(30,cumulative=False)
plt.show()

### stop words

stop_words=set(stopwords.words("English"))
type(stop_words)
print(stop_words)

filtered_words=[]
for w in tokenized_word:
    if w not in stop_words:
        filtered_words.append(w)
    else:
        print("These words were filtered out",w)

print("Tokenized words",filtered_words)



