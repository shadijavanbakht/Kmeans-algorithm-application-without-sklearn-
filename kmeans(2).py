#importing libraries
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from scipy.spatial import distance
import numpy as np
import random
import string
import math
loc=[]
movie_critics = pd.read_csv(r(input('enter your data path as a string')), engine='c')

#tokenizing texts
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    tokens = [word for word in tokens if word not in string.punctuation]
    return tokens
tokenized_critics=[]
for index, row in movie_critics.iterrows():  
    tokenized_text = preprocess_text(row.iloc[0])
    if tokenized_text:
        tokenized_critics.append(tokenized_text)
vectors = Word2Vec(sentences=tokenized_critics, vector_size=100, window=5, min_count=1, workers=4)
def get_vector(word):
    try:
        return vectors.wv[word]
    except KeyError:
        return [0] * 100 
for word in vectors.wv.index_to_key:
    loc.append(get_vector(word))
import random

def euclidean_distance(list1, list2):
    if len(list1) != len(list2):
        raise ValueError(list1,list2)

    sum_of_squares = sum((x - y) ** 2 for x, y in zip(list1, list2))

    distance = math.sqrt(sum_of_squares)

    return distance

mylist=loc
lis1=[]
lis2=[]


def centercheck(center1,center2,center1old,center2old):
    if euclidean_distance(center1,center1old)==0 and euclidean_distance(center2,center2old)==0:
        return False
        
    return True
center1old=[]
center2old=[]
for i in range(100):
    center1old.append(0)
    center2old.append(0)
center1=[]
center2=[]
for i in range(100):
    center1.append(0.3)
    center2.append(-0.3)
count=1
while centercheck(center1,center2,center1old,center2old):
    for i in mylist:
        if euclidean_distance(i,center1)<euclidean_distance(i,center2):
            lis1.append(i)
        else:lis2.append(i)
    center1old=center1
    center2old=center2
    transposed_lists1 = list(map(list, zip(*lis1)))
    averages1 = [sum(lst) / len(lst) for lst in transposed_lists1]
    center1=averages1
    transposed_lists2 = list(map(list, zip(*lis2)))
    averages2 = [sum(lst) / len(lst) for lst in transposed_lists2]
    center2=averages2
    print(count)
    count+=1
    lis1temp, lis2temp= lis1, lis2
    lis1, lis2= [], []


positive = [np.where((np.array(mylist) == element).all(axis=1))[0][0] for element in lis1temp]
negative=[np.where((np.array(mylist) == element).all(axis=1))[0][0] for element in lis2temp]
print('positives are:',positive)
print('-------------------------------------------------------------------------------------')
print('negatives are:',negative)