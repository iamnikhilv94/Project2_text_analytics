import pyodbc          # To connect to the SQL Server
import pandas as pd    # To store the data in a data frame

#Connect to the SQL Server - change Server, database information
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=stwssbsql02;'
                      'Database=FAA1920_G2;'
                      'Trusted_Connection=yes;')

#List filename to copy from SQL Server
filestr = [" FAA_FY2017_201611_NEW"," FAA_FY2017_201612_NEW",]

#Store the first file from the server into data to create a dataframe with a structure
Data = pd.read_sql('SELECT * FROM FAA1920.dbo. FAA_FY2017_201610_NEW',conn)
 
#Run a loop to copy the file from the server and append it into 1 single file
for Filename in filestr:
    #print(Filename)# 
    query = "SELECT * FROM FAA1920.dbo.%s"%(Filename)
    #print(query)#
    Data = Data.append(pd.read_sql(query,conn))




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

#checking and removing Null comments
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

# Data is a dataframe with Division, comments and words
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

#creating a term document matrix
counts.to_csv('Term_document.csv',index=True)

#######################################
##Perfomring a Topic modeling using LDA
#######################################


data=pd.read_csv('text_analytics.csv')

data['index']=data.index
data.head()


text_ama500= data[data['Division']=="AMA-500"]
text_ama500.shape
text_ama500[:3]

text_ama500['index']=range(1,len(text_ama500)+1)
documents=text_ama500
documents = documents.dropna(subset=['Comments'])



import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
stemmer = SnowballStemmer('english')
from nltk.stem.porter import *
np.random.seed(2018)
import nltk
nltk.download('wordnet')

#fucntion to perform lemmatize and stem pre-processing steps


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

doc_sample=documents[documents['index']==1].values[0][1]
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))

#running this on the whole document

processed_docs=documents['Comments'].map(preprocess)
processed_docs[:1]
processed_docs.shape


dictionary=gensim.corpora.Dictionary(processed_docs)

count=0
for k, v in dictionary.iteritems():
    print(k,v)
    count=count+1
    if count>10:
        break

    

dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
bow_corpus[4310]

bow_doc_4310 = bow_corpus[4310]

for i in range(len(bow_doc_4310)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_4310[i][0], dictionary[bow_doc_4310[i][0]], bow_doc_4310[i][1]))


#### Using TFIDF #####

from gensim import corpora, models

tfidf = models.TfidfModel(bow_corpus)

corpus_tfidf = tfidf[bow_corpus]
from pprint import pprint

for doc in corpus_tfidf:
    pprint(doc)
    break
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=4, id2word=dictionary, passes=2, workers=2)

for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))
    

## Running LDA using TFIDF
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=4, id2word=dictionary, passes=2, workers=4)


for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))


processed_docs.iloc[[4310]]

for index, score in sorted(lda_model[bow_corpus[4310]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))

