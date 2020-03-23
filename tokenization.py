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
#displays the number of comments for each division

plt.title('Number of comments in each Division')
data['Division'].value_counts().plot.bar(color='Green')
plt.xlabel('Division')
plt.ylabel('Number of comments')

data.info()
data.groupby('Division').count()

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer

# token=RegexpTokenizer(r'[a-zA-Z0-9+')
# type(token)
# cv=CountVectorizer(lowercase=True,stop_words='english',ngram_range=(1,1),tokenizer=token.tokenize)
# text_counts=cv.fit_transform(data.words)

data['words']=data.Comments.str.strip().str.split('[\W_]+')
data.head()
data.Comments[0]

##creating iterations to flatten the array of words 

rows=list()
for row in data[['Division','words']].iterrows():
    r=row[1]
    for word in r.words:
        rows.append((r.Division,word))

#words dataframe contains Divison and Words from comments    
words=pd.DataFrame(rows,columns=['Division','word'])
words.head()
words.shape

#removing the spaces from word list
words = words[words.word.str.len()>0]

#converting all words to lower case
words['word']=words.word.str.lower()
words.shape
words.info()

#importing the stopwords using scikit feautre extraction

from sklearn.feature_extraction import text
my_stopwords={'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','st'}
stop=text.ENGLISH_STOP_WORDS.union(my_stopwords)

a=pd.DataFrame(words)
a.info()
#removing stopwords from the list of words
a['word']=a['word'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
a.shape
a.info()

counts=a.groupby('Division')\
    .word.value_counts()\
        .to_frame()\
            .rename(columns={'word':'count_words'})

counts.head()
type(counts)

counts.to_csv('Term_document.csv',index=True)



