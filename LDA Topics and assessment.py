import os
import pandas as pd

os.getcwd()
os.chdir('/Users/nikhilviswanath/Documents/python_data/')


data=pd.read_csv('text_analytics.csv')
data1=data[data['Division']=="AMA-20"]
data1=data1.drop(columns=['Division'])
data1['index']=data1.index

data1.shape
data1[:5]

#### Data Pre processing ###

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
stemmer=SnowballStemmer("english")
from nltk.stem.porter import *
import numpy as np
import nltk
nltk.download('wordnet')

#print(WordNetLemmatizer().lemmatize('went',pos='v'))


#stemmer = SnowballStemmer('english')
#original_words = ['knowledge','knowlege']
#singles = [stemmer.stem(plural) for plural in original_words]
#pd.DataFrame(data = {'original word': original_words, 'stemmed': singles})

# Add new stopwords if needed
#stopwords = stopwords.union(set(["add_term_1", "add_term_2"]))


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text,pos='v'))

def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result
       
# CHekcing how stemming and lemmatize is working on sample comments
# =============================================================================
# doc_sample=data1[data1['index']==0].values[0][0]
# print('orginal comment:')
# words=[]
# for word in doc_sample.split(' '):
#     words.append(word)
# print (words)
# print('\n\n tokenized and lemmatized comments:')
# print(preprocess(doc_sample))

data1['processed_comments']=data1['Comments'].fillna('').astype(str).map(preprocess)
processed_docs=data1['Comments'].fillna('').astype(str).map(preprocess)

#Bag of Words
dictionary=gensim.corpora.Dictionary(processed_docs)
count=0
for k,v in dictionary.iteritems():
    print(k,v)
    count+=1
    if count >10:
        break

dictionary.filter_extremes(no_below=15,no_above=0.8,keep_n=1000)
bow_corpus= [dictionary.doc2bow(doc) for doc in processed_docs]

bow_corpus[0]
bow_doc_0 = bow_corpus[0]
for i in range(len(bow_doc_0)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_0[i][0], 
                                                     dictionary[bow_doc_0[i][0]], 
                                                     bow_doc_0[i][1]))


#Running LDA model using Bag of Words

lda_model=gensim.models.LdaMulticore(bow_corpus,
                                    num_topics=3,
                                    id2word=dictionary,
                                    passes=2,
                                    workers=2)

for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx,topic))

help(gensim.models.LdaMulticore)

lda_model[bow_corpus[457]]


# Eval via coherence scoring


from gensim.models import CoherenceModel

coh = CoherenceModel(model=lda_model,
                     texts= processed_docs,
                     dictionary = dictionary,
                     coherence = "c_v")

coh_lda = coh.get_coherence()
print("Coherence Score:", coh_lda)

# Visualizing the clusters

import pyLDAvis.gensim as pyldavis
import pyLDAvis
lda_display = pyldavis.prepare(lda_model, bow_corpus, dictionary)
pyLDAvis.show(lda_display)

##### Creating a feautre vector for each comment using LDA model

test_vecs = []
for i in range(len(bow_corpus)):
    top_topics = lda_model.get_document_topics(bow_corpus[i], minimum_probability=0.0)
    topic_vec = [top_topics[i][1] for i in range(3)]
    test_vecs.append(topic_vec)




############################################################
#################### TF IDF ################################ 
############################################################
# =============================================================================
#     
# from gensim import corpora, models
# tfidf = models.TfidfModel(bow_corpus)
# 
# corpus_tfidf=tfidf[bow_corpus]
# 
# from pprint import pprint
# for doc in corpus_tfidf:
#     print(doc)
#     break
# 
# 
# lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf,
#                                              num_topics=3,
#                                              id2word=dictionary,
#                                              passes=2,
#                                              workers=4)
# 
# for idx, topic in lda_model_tfidf.print_topics(-1):
#     print('Topic: {} Word: {}'.format(idx, topic))
# =============================================================================


















































