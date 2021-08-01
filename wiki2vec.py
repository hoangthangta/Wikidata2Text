#........................................................................................................
# Title: Wikidata claims (statements) to natural language (a part of Triple2Text/Ontology2Text task)
# Author: Ta, Hoang Thang
# Email: tahoangthang@gmail.com
# Lab: https://www.cic.ipn.mx
# Date: 12/2019
#........................................................................................................

from gensim.test.utils import datapath
from gensim.models import KeyedVectors
import gensim.models

# use wiki-news-300d-1M.vec
#import time
#start_time = time.time()
#print("--- %s seconds ---" % (time.time() - start_time))

def get_common_word_list(word_list, word_vectors):
    final_list = []
    for w in word_list:
        try:
            final_list += word_vectors.most_similar(w, topn = 100)
        except:
            continue

    word_dict = {} 
    for w in final_list:
        if w[0] not in word_dict:
            word_dict[w[0]] = w[1]
        else:
            if (w[1] > word_dict[w[0]]): #get the highest distance
                word_dict[w[0]] = w[1]

    '''for key, value in word_dict.items():
        word_dict[key] = sum(value)/len(value)'''

    sorted_list = sorted(word_dict.items(), key=lambda kv: kv[1], reverse = True)
    return sorted_list

def check_words_wiki2vec(words, word_list):
    for word in words:
        if (word not in word_list):
            return False
    return True

def check_word_wiki2vec(word, word_list):
    if (word not in word_list):
        return False
    return True
    
# it may take a long time to finish
def get_wiki2vec_related_words(corpus_path, input_list, limit):
    model = load_wiki2vec(corpus_path, input_list, limit)
    word_distance = get_common_word_list(input_list, model)
    return word_distance

# load wiki2vec, limit should be 500000
def load_wiki2vec(corpus_path, limit):
    #corpus_path = "wiki-news-300d-1M.vec"
    model = gensim.models.KeyedVectors.load_word2vec_format(corpus_path, binary=False, limit = limit)
    return model


