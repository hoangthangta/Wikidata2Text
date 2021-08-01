#........................................................................................................
# Title: Wikidata claims (statements) to natural language (a part of Triple2Text/Ontology2Text task)
# Author: Ta, Hoang Thang
# Email: tahoangthang@gmail.com
# Lab: https://www.cic.ipn.mx
# Date: 12/2019
#........................................................................................................

from base import *

#from cluster_methods import *

from wiki_core import *
from read_write_file import *
from word2vec import *
from wiki2vec import *
from itertools import cycle
from collections import Counter

from sklearn.cluster import DBSCAN, OPTICS, MeanShift, AffinityPropagation, AgglomerativeClustering, SpectralClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import TruncatedSVD, PCA, NMF, SparsePCA, FastICA
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

import numpy as np
import math
import re
import time

import matplotlib.pyplot as plt

from collections import Counter
import pandas as pd

from nltk import ngrams

import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load("en_core_web_md")

definition_properties = {
  'P26': ['spouse', 'wife', 'married', 'marry', 'marriage', 'partner', 'wedded', 'wed', 'wives', 'husbands', 'spouses', 'husband'],
  'P39': ['position', 'political', 'object', 'seat', 'public', 'office', 'subject', 'formerly', 'holds', 'currently', 'held', 'occupied'],
  'P54': ['sports', 'teams', 'clubs', 'member', 'team', 'played', 'plays', 'club', 'player'],
  'P69': ['educated', 'educational', 'institution', 'attended', 'subject', 'alma', 'mater', 'education',
    'alumni', 'alumnus', 'alumna', 'college', 'university', 'school', 'studied', 'graduate', 'graduated', 'faculty'],
  'P108': ['employer', 'person', 'organization', 'subject', 'works', 'worked', 'workplace', 'employed', 'working', 'place'],
  'P166': ['work', 'awarded', 'won', 'medals', 'creative', 'person', 'awards', 'win', 'winner', 'honors', 'received', 'award',
    'prize', 'title', 'recognition', 'honorary', 'honours', 'organisation'],
  'P6': ['first', 'chancellor', 'prime', 'minister', 'government', 'mayor', 'state', 'executive', 'town', 'national', 'other', 'power', 'head', 'municipality', 'country', 'premier', 'body', 'governor', 'heading', 'city', 'headed', 'governmental', 'president'],
  'P17': ['human', 'host', 'used', 'beings', 'item', 'sovereign', 'country', 'nation', 'land', 'state'],
  'P22': ['daughter', 'daddy', 'subject', 'dad', 'male', 'stepfather', 'stepparent', 'son', 'father', 'child', 'parent'],
  'P27': ['national', 'subject', 'nationality', 'country', 'citizenship', 'object', 'citizen', 'recognizes'],
  'P31': ['example', 'main', 'member', 'class', 'individual', 'unsubclassable', 'subject', 'instance', 'occurrence', 'unsubtypable', 'distinct', 'uninstantiable', 'non', 'specific', 'unique', 'rdf', 'unsubclassifiable', 'element', 'unitary', 'type', 'particular'],
  'P35': ['highest', 'king', 'authority', 'governor', 'queen', 'chief', 'monarch', 'head', 'official', 'country', 'headed', 'emperor', 'leader', 'formal', 'state', 'president'],
  'P101': ['specialism', 'specialization', 'speciality', 'studies', 'FOW', 'organization', 'field', 'researcher', 'area', 'work', 'fields', 'person', 'academic', 'research', 'occupation', 'activity', 'subject', 'domain', 'scientific', 'discipline', 'responsible', 'conduct', 'see', 'study'],
  'P103': ['languages', 'mother', 'person', 'language', 'native', 'learned', 'first', 'L1', 'speaker', 'tongue', 'birth'],
  'P106': ['work', 'position', 'held', 'person', 'employment', 'craft', 'occupation', 'profession', 'career', 'field', 'job'],
  'P108': ['person', 'employed', 'workplace', 'employer', 'working', 'works', 'subject', 'worked', 'place', 'organization'],
  'P131': ['district', 'administrative', 'arrondissement', 'rural', 'territorial', 'entity', 'happens', 'village', 'region', 'following', 'territory', 'item', 'Indian', 'local', 'shire', 'government', 'area', 'based', 'borough', 'department', 'state', 'reservation', 'town', 'commune', 'unit', 'places', 'province', 'reserve', 'municipality', 'settlement', 'ward', 'county', 'prefecture', 'non', 'locations', 'parish', 'items', 'principal', 'location', 'voivodeship', 'locality', 'specifying', 'city', 'events', 'located'],
  'P155': ['comes', 'offices', 'prequel', 'preceding', 'prior', 'replaces', 'split', 'sequel', 'item', 'successor', 'immediately', 'follows', 'before', 'series', 'subject', 'replaced', 'political', 'use', 'preceded', 'part', 'succeeds', 'previous', 'predecessor'],
  'P156': ['comes', 'offices', 'prequel', 'part', 'sequel', 'following', 'item', 'successor', 'succeeded', 'immediately', 'followed', 'before', 'preceeds', 'series', 'subject', 'precedes', 'replaced', 'political', 'next', 'use', 'succeded'],
  'P184': ['PhD', 'supervisor', 'doctorate', 'thesis', 'promotor', 'advisor', 'subject', 'doctoral', 'supervised'],
  'P276': ['administrative', 'case', 'entity', 'region', 'physical', 'venue', 'event', 'item', 'place', 'area', 'based', 'object', 'held', 'feature', 'neighborhood', 'distinct', 'origin', 'terrain', 'location', 'use', 'located', 'moveable'],
  'P407': ['broadcasting', 'audio', 'signed', 'URL', 'written', 'languages', 'associated', 'used', 'such', 'name', 'language', 'native', 'available', 'text', 'website', 'work', 'creative', 'named', 'reference', 'spoken', 'websites', 'songs', 'persons', 'use', 'shows', 'books'],
  'P413': ['specialism', 'position', 'played', 'player', 'speciality', 'team', 'fielding'],
  'P453': ['filled', 'specific', 'played', 'cast', 'subject', 'role', 'use', 'plays', 'acting', 'qualifier', 'character', 'member', 'actor', 'only', 'voice'],
  'P512': ['person', 'academic', 'degree', 'holds', 'diploma'],
  'P570': ['date', 'died', 'dead', 'subject', 'deathdate', 'year', 'death', 'end', 'DOD'],
  'P571': ['introduced', 'commenced', 'defined', 'commencement', 'existence', 'point', 'came', 'time', 'creation', 'formation', 'first', 'inception', 'founded', 'written', 'founding', 'built', 'created', 'constructed', 'foundation', 'when', 'inititated', 'date', 'dedication', 'subject', 'establishment', 'issue', 'start', 'inaugurated', 'launch', 'introduction', 'launched', 'formed', 'construction', 'year', 'incorporated', 'incorporation', 'completed', 'established'],
  'P577': ['work', 'date', 'airdate', 'dop', 'released', 'point', 'air', 'initial', 'pubdate', 'time', 'publication', 'first', 'year', 'published', 'release', 'when'],
  'P580': ['start', 'starting', 'began', 'statement', 'date', 'introduced', 'introduction', 'begins', 'item', 'started', 'beginning', 'exist', 'time', 'valid', 'starttime', 'starts', 'building'],
  'P582': ['ending', 'ceases', 'indicates', 'divorced', 'cease', 'left', 'time', 'closed', 'end', 'endtime', 'operation', 'item', 'date', 'stop', 'statement', 'office', 'dissolved', 'ends', 'stops', 'valid', 'being', 'exist', 'fall', 'completed'],
  'P585': ['date', 'statement', 'event', 'existed', 'point', 'something', 'place', 'true', 'time', 'year', 'took', 'when'],
  'P642': ['stating', 'statement', 'item', 'scope', 'qualifier', 'applies', 'particular'],
  'P669': ['road', 'add', 'street', 'square', 'item', 'number', 'where', 'use', 'address', 'qualifier', 'there', 'property', 'located'],
  'P708': ['church', 'types', 'archdiocese', 'division', 'administrative', 'other', 'diocese', 'ecclesiastical', 'use', 'entities', 'bishopric', 'belongs', 'element', 'territorial', 'archbishopric'],
  'P735': ['forename', 'family', 'Christian', 'person', 'used', 'names', 'middle', 'values', 'name', 'should', 'link', 'first', 'disambiguations', 'property', 'given', 'personal'],
  'P748': ['person', 'appointed', 'used', 'can', 'office', 'qualifier'],
  'P768': ['district', 'seat', 'electoral', 'area', 'candidacy', 'representing', 'held', 'Use', 'election', 'riding', 'person', 'office', 'ward', 'position', 'being', 'contested', 'qualifier', 'constituency', 'electorate'],
  'P805': ['dedicated', 'identified', 'statement', 'qualifying', 'item', 'describes', 'subject', 'artfor', 'article', 'relation', 'claim'],
  'P811': ['college', 'someone', 'studied', 'academic', 'minor', 'university'],
  'P812': ['college', 'someone', 'studied', 'academic', 'major', 'subject', 'university', 'field', 'study'],
  'P828': ['due', 'causes', 'has', 'result', 'ultimate', 'had', 'why', 'ultimately', 'implied', 'thing', 'reason', 'effect', 'underlying', 'outcome', 'resulted', 'originated', 'caused', 'cause', 'initial'],
  'P937': ['work', 'workplace', 'persons', 'working', 'activity', 'where', 'location', 'place', 'active'],
  'P1001': ['value', 'institution', 'has', 'territorial', 'jurisdiction', 'item', 'linked', 'law', 'applied', 'state', 'statement', 'office', 'power', 'country', 'municipality', 'valid', 'belongs', 'applies', 'public'],
  'P1013': ['respect', 'basis', 'used', 'according', 'made', 'criterion', 'reference', 'criteria', 'respectively', 'property', 'distinction', 'based', 'classification', 'by'],
  'P1066': ['pupil', 'master', 'person', 'academic', 'disciple', 'supervisor', 'teacher', 'professor', 'studied', 'has', 'mentor', 'advisor', 'taught', 'student', 'tutor'],
  'P1264': ['applicability', 'statement', 'validity', 'period', 'time', 'valid', 'applies', 'when'],
  'P1268': ['represents', 'entity', 'organization', 'organisation', 'individual'],
  'P1350': ['pitched', 'number', 'played', 'races', 'games', 'matches', 'team', 'appearances', 'caps', 'starts', 'gp', 'sports', 'mp'],
  'P1351': ['scored', 'used', 'event', 'number', 'league', 'participant', 'points', 'goals', 'qualifier', 'GF', 'score', 'set', 'match', 'use'],
  'P1365': ['replaces', 'structures', 'identical', 'item', 'successor', 'continues', 'forefather', 'follows', 'holder', 'person', 'job', 'replaced', 'structure', 'preceded', 'supersedes', 'succeeds', 'previous', 'predecessor'],
  'P1366': ['adds', 'role', 'identical', 'item', 'heir', 'successor', 'succeeded', 'continues', 'superseded', 'followed', 'dropping', 'holder', 'person', 'other', 'series', 'job', 'replaced', 'next', 'replacing', 'continued', 'mediatised', 'books'],
  'P1534': ['date', 'ending', 'specify', 'together', 'use', 'qualifier', 'cause', 'ended', 'end', 'reason'],
  'P1642': ['status', 'transaction', 'player', 'acquisition', 'acquired', 'team', 'how', 'qualifier', 'member', 'loan', 'contract', 'sports'],
  'P1686': ['work', 'awarded', 'nominated', 'received', 'award', 'qualifier', 'citation', 'creator', 'given'],
  'P1706': ['item', 'together', 'award', 'tied', 'feat', 'qualifier', 'featuring', 'property', 'shared', 'accompanied', 'specify'],
  'P2389': ['leads', 'person', 'directed', 'office', 'head', 'heads', 'directs', 'leader', 'organization', 'runs', 'organisation', 'led'],
  'P2578': ['learning', 'research', 'academic', 'item', 'working', 'study', 'subject', 'studies', 'researches', 'property', 'object', 'field', 'studying', 'scholarly'],
  'P2715': ['election', 'position', 'reelection', 'confirmed', 'person', 'statements', 'gained', 'qualifier', 'link', 'elected', 'held'],
  'P2842': ['wedding', 'location', 'where', 'place', 'spouse', 'celebrated', 'marriage', 'property', 'married'],
  'P2868': ['value', 'duty', 'function', 'context', 'has', 'role', 'title', 'purpose', 'generic', 'item', 'acting', 'identity', 'character', 'object', 'statement', 'subject', 'roles', 'job', 'use'],
  'P3831': ['value', 'generic', 'statement', 'context', 'specifically', 'circumstances', 'item', 'employment', 'subject', 'role', 'identity', 'use', 'qualifier', 'object'],
  'P4100': ['parliament', 'group', 'faction', 'belongs', 'parliamentary', 'member', 'party'],
  'P1319': ['date', 'earliest']
}

# load corpus from property name
def load_corpus(file_name, word2vec_file_name, property_name, delimiter='#', dtype=dtypes, trained=False, idf_dict_status=False):

    df = pd.read_csv(file_name, delimiter='#', dtype=dtype, usecols=list(dtype))
    best_sentences, best_rows = get_best_sentences(df, show=False)
    labeled_sen_list = df['labeled_sentence_2']
    
    counter = create_ngram(labeled_sen_list, 1) # unigram

    idf_dict = {}
    if (idf_dict_status == True):
        idf_dict = create_idf_dict(labeled_sen_list)
    
    word_corpus = create_true_distribution_corpus2(labeled_sen_list, 0)

    if (trained == True):
        word2vec_train(word2vec_file_name, property_name, word_corpus)

    # load models   
    local_model = load_word2vec(word2vec_file_name)
    global_model = load_wiki2vec('D:\wiki-news-300d-1M.vec', 200000)

    result_dict = {}
    result_dict['file_name'] = file_name
    result_dict['sen_list'] = df
    result_dict['best_sentences'] = best_sentences
    result_dict['labeled_sen_list'] = labeled_sen_list
    result_dict['counter'] = counter
    result_dict['idf_dict'] = idf_dict
    result_dict['word_corpus'] = word_corpus
    result_dict['local_model'] = local_model
    result_dict['global_model'] = global_model

    print('Loading corpus was done!!!')

    return result_dict

# some basic statistics
def basic_statistics(file_name, delimiter='#', dtype=dtypes, best_sentence = False):
    print('file_name: ', file_name)
    #sen_list = read_from_csv_file(file_name,  '#', 'all')[1:] # remove headers
    df = pd.read_csv(file_name, delimiter=delimiter, dtype=dtype, usecols=list(dtype))

    average_sentence_length(df)
    average_word(df)
    average_token(df)
    average_token_labeled_sentence(df) 
    ratio_token_per_quad(df)
    ratio_token_per_quad_item(df)

    if (best_sentence == True):
        print('++ Best sentences statistics')
        labeled_list, df2 = get_best_sentences(df)
        #print(len(labeled_list))
        
        average_sentence_length(df2)
        average_word(df2)
        average_token(df2)
        average_token_labeled_sentence(df2) 
        ratio_token_per_quad(df2)
        ratio_token_per_quad_item(df2)
    print('.............................')
    print('.............................')


# cumulative rate by property
def cumulative_rate_by_property(property_name, df):


    length_list  = []
    for index, row in df.iterrows():
        #print(property_name, row['length'])

        if (row['predicate'].lower() == property_name.lower()):
            length_list.append(int(row['length']))

        elif (property_name == 'common'): # count all properties
            length_list.append(int(row['length']))
            

    #file_name, sen_list, best_sentences, labeled_sen_list, counter, idf_dict, word_corpus, local_model, global_model = load_corpus(property_name)
    #sentences, number_redundant_word_list, redundant_word_list = get_corpus_redundant_words(sen_list)

    #print('length_list: ', length_list)
    rank_list = rank_sentence_by_redundant_words(length_list)
    cumulative_list = cumulative_rate(rank_list)
    #print('rank_list: ', rank_list)
    
    return cumulative_list

def treat_labeled_items2():
    prefixes = list(nlp.Defaults.prefixes)
    prefixes.remove('\\[')

    prefix_regex = spacy.util.compile_prefix_regex(prefixes)
    nlp.tokenizer.prefix_search = prefix_regex.search

    suffixes = list(nlp.Defaults.suffixes)
    suffixes.remove('\\]')

    suffix_regex = spacy.util.compile_suffix_regex(suffixes)
    nlp.tokenizer.suffix_search = suffix_regex.search

    infixes = list(nlp.Defaults.prefixes)
    infixes.remove('\\[')
    infixes.remove('\\]')

    try:
        infixes.remove('\\-')
    except Exception as e:
        pass

    try:
        infixes.remove(':')
    except Exception as e:
        pass

    try:
        infixes.remove('_')
    except Exception as e:
        pass

    infix_regex = spacy.util.compile_infix_regex(infixes)
    nlp.tokenizer = Tokenizer(nlp.vocab, infix_finditer=infix_regex.finditer)

# create n-gram from text
def ngram(text, n):
    # make n-gram and also count the frequency of each item by Counter

    treat_labeled_items2()
    doc = nlp(text)

    temp = [token.text for token in doc if token.text != '']

    return list(ngrams(temp, n))

# create n-gram from list
def create_ngram(sentence_list, n):
    temp = []
    for sentence in sentence_list:
        sentence = "[start] " + sentence  + " [end]"
        temp += (ngram(sentence, n))
    return Counter(temp)

# filter by property name
def filter_by_property(property_name, sen_list):

    #property_list = ['P26','P39','P54','P69','P108','P166']
    result_list = []
    for p in sen_list[1:]: # start with data in line 1 (not headers)
    
        if (p[2] == property_name):
            result_list.append(p)

    result_list = sorted(result_list, key = lambda x: (int(x[2][1:]), x[4])) # sort by qualifier
    return result_list

# write file from list
def write_file_from_list(file_name, sen_list):
    with open(file_name,'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f, delimiter='#', quoting=csv.QUOTE_MINIMAL)

        for p in sen_list:
            print(p)
            wr.writerow(p)

# average length per raw sentence
def average_sentence_length(df):
    al = 0

    for index, row in df.iterrows():
        #print('row: ', row)
        al += len(row['raw_sentence'])

    print('average_sentence_length: ', al/len(df))
    return al/len(df)

# average word per raw sentence
def average_word(df):
    al = 0

    for index, row in df.iterrows():
        doc = nlp(row['raw_sentence'])
        # words = [token.text for token in doc if token.is_punct != True]
        al += len(row['raw_sentence'].split())

    print('average_word: ', al/len(df))
    return al/len(df)

# average token per raw sentence
def average_token(df):
    al = 0

    for index, row in df.iterrows():
        doc = nlp(row['raw_sentence'])
        al += doc.__len__()
        
    print('average_token: ', al/len(df))
    return al/len(df)

# average token per labeled sentence
def average_token_labeled_sentence(df):
    al = 0
    treat_labeled_items() # treat a labeled item as a token

    for index, row in df.iterrows():
        doc = nlp(row['labeled_sentence_2'])      
        al += doc.__len__()

    print('average_token_labeled_sentence: ', al/len(df))
    return al/len(df)

# ratio of token per quad (labeled sentence)
def ratio_token_per_quad(df):
    treat_labeled_items() # treat a labeled item as a token
    tokens = 0
    quads = len(df) # 1 quad in 1 sentence

    for index, row in df.iterrows():
        doc = nlp(row['labeled_sentence_2'])
        tokens += doc.__len__()

    print('ratio_token_per_quad: ', tokens/quads)
    return tokens/quads

# ratio of token per quad item (labeled sentence)
def ratio_token_per_quad_item(df):
    treat_labeled_items() # treat a labeled item as a token
    tokens = 0
    quad_items = 0
    
    for index, row in df.iterrows():
        doc = nlp(row['labeled_sentence_2'])
        temp_quads = len(row['order_2'].split(','))
        tokens += doc.__len__() - temp_quads
        quad_items += temp_quads

    print('ratio_token_per_quad_item: ', tokens/quad_items)
    return tokens/quad_items

# get the best sentences: no redundant words (except stop words & a verb as ROOT)
def get_best_sentences(df, show=False):

    treat_labeled_items2() # treat a labeled item as a token
    best_sentence_list = []
    best_row_list = []

    columns = []
    if (len(df) != 0):
        columns = [index for index, val in df.iloc[0].iteritems()]

    for index, row in df.iterrows():
        doc = nlp(row['labeled_sentence_2']) 

        redudant_list = []
        temp_quads = [x.strip() for x in row['order_2'].split(',')]

        for token in doc:
            if (token.pos_ == "X"):
                continue
            if (token.pos_ == "PUNCT"):
                continue
            if (token.pos_ == "CCONJ"):
                continue
            if (token.pos_ == "ADP"):
                continue
            if (token.pos_ == "PRON"):
                continue
            if (token.pos_ == "PART"):
                continue
            if (token.pos_ == "DET"):
                continue
            if (token.dep_ == "punct"):
                continue

            if (token.text not in temp_quads):
                redudant_list.append([token.text, token.pos_, token.dep_])
                #print(token.text, token.pos_, token.dep_)

        if (len(redudant_list) == 1):
            if (redudant_list[0][2] == "ROOT"): # token.pos_
                if (row['labeled_sentence_2'] not in best_sentence_list):
                    best_sentence_list.append(row['labeled_sentence_2']) # add the labeled sentence only
                    best_row_list.append([val for index, val in row.iteritems()]) # add a whole row
                        

    if (show != False):
        print('..............................')
        print('..............................')
        print('Best sentences:')
        for s in best_sentence_list:
            print(s)
            print('-----------')
        print('..............................')
        print('..............................')

    # convert to dataframe
    df = pd.DataFrame(best_row_list, columns=columns)
    #print('df: ', df)
    
    return best_sentence_list, df

# get redundant words in labeled sentences
def get_redundant_words(sen_row):

    redudant_list = []
    treat_labeled_items2()
    doc = nlp(sen_row['labeled_sentence_2'])

    quad_items = get_quad_items(sen_row)

    for token in doc:
        if (token.pos_ == "X"):
            continue
        if (token.pos_ == "PUNCT"):
            continue
        if (token.pos_ == "CCONJ"):
            continue
        if (token.pos_ == "ADP"):
            continue
        if (token.pos_ == "PRON"):
            continue
        if (token.pos_ == "PART"):
            continue
        if (token.pos_ == "DET"):
            continue
        if (token.dep_ == "punct"):
            continue

        if (token.text not in quad_items and token.text.strip() != ''):
            #redudant_list.append([token.text, token.pos_, token.dep_])
            redudant_list.append(token.text)

    return redudant_list

# train corpus using CBOW
def word2vec_train(word2vec_file, property_name, corpus):
    
    # save_word2vec(corpus, min_count, size, window, sorted_vocab, sg, workers, iters, file_name)
    if (property_name == 'p26'):
        save_word2vec(corpus, 0, 150, 2, 1, 0, 8, 10, word2vec_file)

    if (property_name == 'p108'):
        save_word2vec(corpus, 0, 150, 2, 1, 0, 8, 30, word2vec_file)

    if (property_name == 'p69'):
        save_word2vec(corpus, 0, 150, 2, 1, 0, 8, 20, word2vec_file)

    if (property_name == 'p166'):
        save_word2vec(corpus, 0, 150, 2, 1, 0, 8, 20, word2vec_file)

    if (property_name == 'p54'):
        save_word2vec(corpus, 0, 150, 2, 1, 0, 8, 5, word2vec_file)

    if (property_name == 'p39'):
        save_word2vec(corpus, 0, 150, 2, 1, 0, 8, 25, word2vec_file)

    if (property_name == 'common'):
        save_word2vec(corpus, 0, 150, 2, 1, 0, 8, 3, word2vec_file)
        
    if (property_name == 'common2'):
        save_word2vec(corpus, 0, 150, 2, 1, 0, 8, 3, word2vec_file)


# get quad items in a sentence
def get_quad_items(sen_row):
    quad_items = []
    quad_items = [x.strip() for x in sen_row['order_2'].split(',')]
    return quad_items

# get important quad items in a sentence
def get_important_quad_items(sen_row):
    quad_items = []
    quad_items = [x.strip() for x in sen_row['order_2'].split(',')]
    quad_items = list(set(quad_items) - set(['[det:the]','[det:a-an]','[s:poss]'])) # remove unimportant terms
    return quad_items

# get qualifier quad items in a sentence
def get_qualifier_items(sen_row):
    quad_items = []
    quad_items = [x.strip() for x in sen_row['order_2'].split(',')]
    qualifier_list = []
    for q in quad_items:
        if ('qualifier' in q and  'o0' in q): # get qualifiers of o0 (main object or first object)
            qualifier_list.append(q)

    #print('qualifier_list: ', qualifier_list)
    return qualifier_list

# convert sentence to measures (tf, idf, local_distance, global_distance, vector, etc) & write to a result file
def convert_sentence_to_measures(output_file_name, sen_row, best_sentences, local_model, global_model, counter, idf_dict):

    #print('sen_row: ', sen_row)
    
    # redundant words
    redundant_words = get_redundant_words(sen_row)
    length = len(redundant_words)

    sentence = redundant_words # check redundant words only

    # best sentence
    label = ''
    if (sen_row['labeled_sentence_2'] in best_sentences): label = 'x'
    
    # sum & product
    tf1, tf2 = convert_sentence_to_tf(sentence, local_model, counter)
    idf1, idf2 = convert_sentence_to_idf(sentence, idf_dict)
    local1, local2 = convert_sentence_to_local_distance(sen_row, sentence, local_model, counter)
    global1, global2 = convert_sentence_to_global_distance(sen_row, sentence, global_model)

    # combination
    tf_idf1, tf_idf2 =  convert_sentence_to_tf_idf(sentence, local_model, counter, idf_dict)
    local_tf1, local_tf2 = convert_sentence_to_local_tf_distance(sen_row, sentence, local_model, counter)
    local_idf1, local_idf2 = convert_sentence_to_local_idf_distance(sen_row, sentence, local_model, counter, idf_dict)
    local_tf_idf1, local_tf_idf2 = convert_sentence_to_local_tf_idf_distance(sen_row, sentence, local_model, counter, idf_dict)

    global_tf1, global_tf2 = convert_sentence_to_global_tf_distance(sen_row, sentence, global_model, counter, qualifier=False)
    global_idf1, global_idf2 = convert_sentence_to_global_idf_distance(sen_row, sentence, global_model, idf_dict, qualifier=False)
    global_tf_idf1, global_tf_idf2 = convert_sentence_to_global_tf_idf_distance(sen_row, sentence, global_model, counter, idf_dict,
                                                                                qualifier=False)
    # global with qualifier
    global_qualifier1, global_qualifier2 = convert_sentence_to_global_distance(sen_row, sentence, global_model, qualifier=True)
    global_qualifier_tf1, global_qualifier_tf2 = convert_sentence_to_global_tf_distance(sen_row, sentence, global_model, counter, qualifier=True)
    global_qualifier_idf1, global_qualifier_idf2 = convert_sentence_to_global_idf_distance(sen_row, sentence, global_model, idf_dict, qualifier=True)
    global_qualifier_tf_idf1, global_qualifier_tf_idf2 = convert_sentence_to_global_tf_idf_distance(sen_row, sentence, global_model, counter, idf_dict,
                                                                                qualifier=True)
    # vector
    vector_sum, vector_product = convert_sentence_to_vector(sentence, local_model) # base on local_model

    # add results to sen_row
    temp_list = [label, redundant_words, length, tf1, tf2, idf1, idf2, local1, local2, global1, global2, tf_idf1, tf_idf2, local_tf1,
                 local_tf2, local_idf1, local_idf2, local_tf_idf1, local_tf_idf2, global_tf1, global_tf2, global_idf1, global_idf2,
                 global_tf_idf1, global_tf_idf2, global_qualifier1, global_qualifier2, global_qualifier_tf1, global_qualifier_tf2,
                 global_qualifier_idf1, global_qualifier_idf2, global_qualifier_tf_idf1, global_qualifier_tf_idf2]

    sen_row = sen_row.values.tolist()
    sen_row.extend(temp_list)    
    write_to_csv_file(output_file_name, '#', sen_row)

# count average distance of a word to other words in a sentence (use important terms)
def convert_sentence_to_global_distance(sen_row, sentence, global_model, qualifier=False):

    predicate = sen_row[2]
    #print('predicate: ', predicate)
    definition_word_list = []
    if (predicate in definition_properties):
        definition_word_list = definition_properties[predicate]

    if (qualifier == True): 
        qualifiers = sen_row[4].split('-')    
        for q in qualifiers:
            if (q in definition_properties):
                definition_word_list += definition_properties[q]
        
    length = len(sentence)
    sum_dist = 0
    product_dist = 1

    def_length = len(definition_word_list)
    if (def_length == 0): return 0, 0

    for w in sentence:
        temp_sum = 0
        for item in definition_word_list:
            sim = 0
            try:
                # raw normalized similarity, change range [-1,1] to [0,1]
                sim = (global_model.similarity(w, item) + 1)/2
                temp_sum += sim

                #print('w, item: ', w, item)
                #print('sim: ', sim)
            except:
                pass

        if (temp_sum == 0): continue

        '''print('temp_sum: ', temp_sum)
        print('def_length: ', def_length)
        print('...............')
        print('...............')'''

        sum_dist += temp_sum/def_length
        product_dist *= -math.log(temp_sum/def_length)

    # return sum_dist, math.log(product_dist + 1)
    return sum_dist, -math.log(product_dist)

# count average distance of a word to other words in a sentence (use important terms)
def convert_sentence_to_global_tf_idf_distance(sen_row, sentence, global_model, counter, idf_dict, qualifier=False):

    predicate = sen_row[2]
    #print('predicate: ', predicate)
    definition_word_list = []
    if (predicate in definition_properties):
        definition_word_list = definition_properties[predicate]

    if (qualifier == True): 
        qualifiers = sen_row[4].split('-')    
        for q in qualifiers:
            if (q in definition_properties):
                definition_word_list += definition_properties[q]
        
    length = len(sentence)
    sum_dist = 0
    product_dist = 1

    def_length = len(definition_word_list)
    if (def_length == 0): return 0, 0

    for w in sentence:
        temp_sum = 0
        for item in definition_word_list:
            sim = 0
            try:
                # raw normalized similarity, change range [-1,1] to [0,1]
                sim = (global_model.similarity(w, item) + 1)/2
                temp_sum += sim

                #print('w, item: ', w, item)
                #print('sim: ', sim)
            except:
                pass

        if (temp_sum == 0): continue

        '''print('temp_sum: ', temp_sum)
        print('def_length: ', def_length)
        print('...............')
        print('...............')'''
        idf = get_idf(idf_dict, w) # inverse topic frequency
        tf = get_tf(counter, w) # term frequency
        sum_dist += (temp_sum*idf*tf)/def_length
        product_dist *= math.log(1 + (temp_sum*idf*tf)/def_length)

    # return sum_dist, math.log(product_dist + 1)
    return sum_dist, math.log(product_dist + 1)

# count average distance of a word to other words in a sentence (use important terms)
def convert_sentence_to_global_idf_distance(sen_row, sentence, global_model, idf_dict, qualifier=False):

    predicate = sen_row[2]
    #print('predicate: ', predicate)
    definition_word_list = []
    if (predicate in definition_properties):
        definition_word_list = definition_properties[predicate]

    if (qualifier == True): 
        qualifiers = sen_row[4].split('-')    
        for q in qualifiers:
            if (q in definition_properties):
                definition_word_list += definition_properties[q]
        
    length = len(sentence)
    sum_dist = 0
    product_dist = 1

    def_length = len(definition_word_list)
    if (def_length == 0): return 0, 0

    for w in sentence:
        temp_sum = 0
        for item in definition_word_list:
            sim = 0
            try:
                # raw normalized similarity, change range [-1,1] to [0,1]
                sim = (global_model.similarity(w, item) + 1)/2
                temp_sum += sim

                #print('w, item: ', w, item)
                #print('sim: ', sim)
            except:
                pass

        if (temp_sum == 0): continue

        '''print('temp_sum: ', temp_sum)
        print('def_length: ', def_length)
        print('...............')
        print('...............')'''
        idf = get_idf(idf_dict, w) # inverse topic frequency
        sum_dist += (temp_sum*idf)/def_length
        product_dist *= math.log(1 + (temp_sum*idf)/def_length)

    # return sum_dist, math.log(product_dist + 1)
    return sum_dist, math.log(product_dist + 1)

# count average distance of a word to other words in a sentence (use important terms)
def convert_sentence_to_global_tf_distance(sen_row, sentence, global_model, counter, qualifier=False):

    predicate = sen_row[2]
    #print('predicate: ', predicate)
    definition_word_list = []
    if (predicate in definition_properties):
        definition_word_list = definition_properties[predicate]

    if (qualifier == True): 
        qualifiers = sen_row[4].split('-')    
        for q in qualifiers:
            if (q in definition_properties):
                definition_word_list += definition_properties[q]
        
    length = len(sentence)
    sum_dist = 0
    product_dist = 1

    def_length = len(definition_word_list)
    if (def_length == 0): return 0, 0

    for w in sentence:
        temp_sum = 0
        for item in definition_word_list:
            sim = 0
            try:
                # raw normalized similarity, change range [-1,1] to [0,1]
                sim = (global_model.similarity(w, item) + 1)/2
                temp_sum += sim

                #print('w, item: ', w, item)
                #print('sim: ', sim)
            except:
                pass

        if (temp_sum == 0): continue

        '''print('temp_sum: ', temp_sum)
        print('def_length: ', def_length)
        print('...............')
        print('...............')'''
        tf = get_tf(counter, w) # term frequency
        sum_dist += (temp_sum*tf)/def_length
        product_dist *= math.log(1 + (temp_sum*tf)/def_length)

        #print('---', (temp_sum*tf)/def_length, math.log((temp_sum*tf)/def_length))

    #print('product_dist: ', product_dist)
    # return sum_dist, math.log(product_dist + 1)
    return sum_dist, math.log(product_dist + 1)

# convert sentence to a vector distance (similarity)
def convert_sentence_to_local_distance(sen_row, sentence, local_model, counter):

    length = len(sentence)
    sum_dist = 0
    product_dist = 1

    quad_items = get_important_quad_items(sen_row)
    quad_length = len(quad_items)

    for w in sentence:
        temp_sum = 0
        for item in quad_items:
            sim = 0
            try:
                # raw normalized similarity, change range [-1,1] to [0,1]
                sim = ((local_model.similarity(w, item) + 1)/2)
                temp_sum += sim
            except:
                pass
        if (temp_sum == 0): continue
        sum_dist += temp_sum/quad_length
        product_dist *= -math.log(temp_sum/quad_length)

    #return sum_dist, math.log(product_dist + 1)
    return sum_dist, -math.log(product_dist)

# convert sentence to local-tf
def convert_sentence_to_local_tf_distance(sen_row, sentence, local_model, counter):

    length = len(sentence)
    sum_dist = 0
    product_dist = 1

    quad_items = get_important_quad_items(sen_row)
    quad_length = len(quad_items)

    for w in sentence:
        temp_sum = 0
        for item in quad_items:
            sim = 0
            try:
                # raw normalized similarity, change range [-1,1] to [0,1]
                sim = ((local_model.similarity(w, item) + 1)/2)
                temp_sum += sim
            except:
                pass
        if (temp_sum == 0): continue

        tf = get_tf(counter, w) # term frequency
        sum_dist += (temp_sum*tf)/quad_length
        product_dist *= math.log(1 + (temp_sum*tf)/quad_length)

        #print('---', (temp_sum*tf)/quad_length, math.log((temp_sum*tf)/quad_length))

    #return sum_dist, math.log(product_dist + 1)

    #print('product_dist: ', product_dist)
    return sum_dist, math.log(product_dist + 1)

# convert sentence to local-idf
def convert_sentence_to_local_idf_distance(sen_row, sentence, local_model, counter, idf_dict):

    length = len(sentence)
    sum_dist = 0
    product_dist = 1

    quad_items = get_important_quad_items(sen_row)
    quad_length = len(quad_items)

    for w in sentence:
        temp_sum = 0
        for item in quad_items:
            sim = 0
            try:
                # raw normalized similarity, change range [-1,1] to [0,1]
                sim = ((local_model.similarity(w, item) + 1)/2)
                temp_sum += sim
            except:
                pass
        if (temp_sum == 0): continue

        idf = get_idf(idf_dict, w) # inverse topic frequency
        sum_dist += (temp_sum*idf)/quad_length
        product_dist *= math.log(1 + (temp_sum*idf)/quad_length)

    #return sum_dist, math.log(product_dist + 1)
    return sum_dist, math.log(product_dist + 1)

# convert sentence to local-tf-idf
def convert_sentence_to_local_tf_idf_distance(sen_row, sentence, local_model, counter, idf_dict):

    length = len(sentence)
    sum_dist = 0
    product_dist = 1

    quad_items = get_important_quad_items(sen_row)
    quad_length = len(quad_items)

    for w in sentence:
        temp_sum = 0
        for item in quad_items:
            sim = 0
            try:
                # raw normalized similarity, change range [-1,1] to [0,1]
                sim = ((local_model.similarity(w, item) + 1)/2)
                temp_sum += sim
            except:
                pass
        if (temp_sum == 0): continue
        tf = get_tf(counter, w) # term frequency
        idf = get_idf(idf_dict, w) # inverse topic frequency
        sum_dist += (temp_sum*tf*idf)/quad_length
        product_dist *= math.log(1 + (temp_sum*tf*idf)/quad_length)

    #return sum_dist, math.log(product_dist + 1)
    return sum_dist, math.log(product_dist + 1)


# convert sentence to tf-idf
def convert_sentence_to_tf_idf(sentence, model, counter, idf_dict):

    length = len(sentence)
    sum_score = 0
    product_score = 1

    for w in sentence:
        try:
            tf = get_tf(counter, w)
            idf = get_idf(idf_dict, w)
            
            sum_score += tf*idf
            product_score *= tf*idf
        except:
            pass

    return sum_score, math.log(product_score + 1)

# convert sentence to term frequency
def convert_sentence_to_tf(sentence, model, counter):

    #length = len(sentence)
    sum_score = 0
    product_score = 1

    for w in sentence:
        try:
            score = get_tf(counter, w)
            #print('---', score)
            sum_score += score
            product_score *= score
        except:
            pass

    #print('product_score: ', product_score)
    return sum_score, math.log(product_score)

# convert sentence to term frequency
def convert_sentence_to_idf(sentence, idf_dict):

    length = len(sentence)
    sum_score = 0
    product_score = 1

    for w in sentence:
        try:
            score = get_idf(idf_dict, w)
            sum_score += score
            product_score *= score
        except:
            pass

    return sum_score, math.log(product_score + 1)

# convert sentence to vector
def convert_sentence_to_vector(sentence, model):

    length = len(sentence)
    sum_vec = 1
    product_vec = 1

    for w in sentence:
        try:
            w_vec = model.get_vector(w)
            sum_vec += w_vec
            product_vec *= w_vec
        except:
            pass

    return sum_vec, product_vec

# convert corpus to vector
def convert_corpus_to_vector(corpus, best_sentences, model, counter):

    label_list = []
    vector_list = []
    i = 0

    # note that a sentence is a list of words
    for sentence in corpus:
        # convert back to a string sentence
        temp = ' '.join(e for e in sentence[1:-1]) # remove [start], [end]

        if (temp in best_sentences):
            label_list.append('x')
        else:
            label_list.append('')

        sum_vector, product_vector = convert_sentence_to_vector(sentence, model, counter)
        vector_list.append([sum_vector, product_vector])
        i = i + 1

    return label_list, vector_list


# get redundant words and their length for all sentences
def get_corpus_redundant_words(sen_list):

    sentence_list, number_redundant_word_list, redundant_word_list = [], [], []
    for p in sen_list:
        redundant_words = get_redundant_words(p)
        length = len(redundant_words)
        number_redundant_word_list.append(length)
        redundant_word_list.append(redundant_words)
        sentence_list.append(p['labeled_sentence_2'])

    return sentence_list, number_redundant_word_list, redundant_word_list


# convert corpus to measures and write to file
def convert_corpus_to_measures(output_file_name, sen_list, best_sentences, local_model, global_model, counter, idf_dict):

    #label_list, metric_list, number_redundant_word_list, redundant_word_list, sentence_list  = [], [], [], [], []
    #i = 0
    
    # note that sentence is a list of words

    for index, sen_row in sen_list.iterrows():
        convert_sentence_to_measures(output_file_name, sen_row, best_sentences, local_model, global_model, counter, idf_dict)

        #metric_list.append(temp_dict)
        #i = i + 1

    #return label_list, metric_list, number_redundant_word_list, redundant_word_list, sentence_list

#...........................
#...........................

# rank a predicate frequency by property (P26, P39, P54, etc)
def rank_predicate_by_property(count_list, property_name):

    # group and calculate average values
    temp_list = []
    for i in count_list:
        if (i['term'].split('-')[0] == property_name):
            temp_list.append([i['term'], i['local_average'], i['local_max_dist'], i['global_average'], i['global_max_dist'],
                              i['subject_dist'], i['object_dist'], i['redundant_words']])
    df = pd.DataFrame(temp_list)
    df = df.groupby([0]).agg('mean')
    df = {x[0]: x[1:] for x in df.itertuples(index=True)}


    # calculate term frequency and add it & average values to freq_dict 
    freq_list = [t[0] for t in temp_list]
    #print('freq_list: ', freq_list)
    length = len(freq_list) # size of corpus  
    freq_dict = Counter(freq_list)
    #print('freq_dict: ', freq_dict)

    
    for k, v in freq_dict.items():
        freq_dict[k] = {'tf':v/length, 'local_average':df[k][0], 'local_max_dist':df[k][1],
                        'global_average': df[k][2], 'global_max_dist':df[k][3], 'subject_dist': df[k][4],
                        'object_dist':df[k][5], 'redundant_words':df[k][6]}

    #print('freq_dict: ', freq_dict)
    return freq_dict

# count the average distance of a word to other words (important words/terms only) in the same sentence 
def word_distance_to_sentence(quad_items, word, local_model, global_model):

    local_length = len(quad_items) # the numbers of quad items
    global_length = 0

    local_sum = 0 
    global_sum = 0 
    
    local_max_dist = 0
    global_max_dist = 0

    subject_dist = object_dist = 0

    try:
        subject_dist = local_model.similarity(word, '[s]') # subject distance
        object_dist = local_model.similarity(word, '[o0]') # object distance
    except:
        pass

    # local model
    for term in quad_items: # can be qualifiers or all items in quad (subject, object, qualifiers)
        try:
            dist = local_model.similarity(word, term)
            #print('dist, word, term: ', dist, word, term)
            if (dist > local_max_dist):
                local_max_dist = dist
            local_sum += dist
            #print('local_sum: ', local_sum)
        except:
            local_length = local_length - 1 # word is not in model
            pass
    
    # global model

    #print('quad_items: +++', quad_items)
    for term in quad_items:
        
        value = term[term.index(':')+1:term.index('-')]

        temp_list = []
        try:
            temp_list = definition_properties[value]
        except:
            pass
        temp_length = len(temp_list)

        #print('term, value, temp_list, temp_length: ', term, value, temp_list, temp_length)

        for t in temp_list:
            try:
                dist = global_model.similarity(word, t)
                #print('dist: ', dist, word, t)
                if (dist > global_max_dist):
                    global_max_dist = dist
                global_sum += dist
            except:
                temp_length = temp_length - 1 # word is not in model
                pass
        global_length += temp_length

    local_average = global_average = 0
    if (local_length == 0): local_average = 0
    else: local_average = local_sum/local_length
    
    if (global_length == 0): global_average = 0
    else: global_average = global_sum/global_length

    result_dict = {'local_average':local_average, 'local_max_dist': local_max_dist,
                   'global_average': global_average, 'global_max_dist': global_max_dist,
                   'subject_dist': subject_dist, 'object_dist': object_dist}

    #print('result_dict: ', result_dict)
    return result_dict

# count average distance of a word to other words in a sentence (use important terms)
def word_distance_to_property_definition(prop_items, word, global_model):

    length = len(prop_items)
    temp_sum = 0
    max_dist = 0

    for term in prop_items:
        try:
            dist = global_model.similarity(word, term)
            if (dist > max_dist):
                max_dist = dist
            temp_sum += dist
        except:
            length = length - 1 # word is not in model
            pass

    if (length == 0):
        return temp_sum, max_dist
    return temp_sum/length, max_dist

# rank predicate (Wikidata properties) by term frequency
def rank_predicate(sen_df, best_sentences, counter, local_model, global_model, by_qualifier=False):

    result_dict = Counter()
    predicate_criteria_list = [] # list of criteria of each predicate
    property_name_list = []

    redundant_list = []

    for index, sen_row in sen_df.iterrows():

        predicate = sen_row['predicate'].strip() # Wikidata property
        qualifiers = sen_row['qualifiers'].strip().split('-')
        #prepositional_verb = sen_row['prepositional_verb'].split(',')[0].strip("'")

        root = sen_row['root'].split(',')
        root_value = root[0].strip("'") # value of root (verb)
        root_pos = root[1] # position of root
        quad_items = get_qualifier_items(sen_row)

        distance_dict = word_distance_to_sentence(quad_items, root_value, local_model, global_model)
        if (by_qualifier == True):
            term = predicate + '-' + root_value + '-' + '-'.join(qualifiers)
        else:
            term = predicate + '-' + root_value
            
        property_name_list.append(predicate)
        distance_dict['term'] = term
        redundant_words = get_redundant_words(sen_row)
        distance_dict['redundant_words'] = len(redundant_words)
        
        predicate_criteria_list.append(distance_dict)
        
    property_names = list(set(property_name_list))

    # join dictionaries by property
    for pn in property_names:
        result_dict = {**result_dict, **rank_predicate_by_property(predicate_criteria_list, pn)} # join two dictionaries

    normalized_values = []
    normalized_labels = []
    
    for k, v in result_dict.items():   
        temp = k.split('-')
        property_name = temp[0]
        predicate = temp[1]
        tf = get_tf(counter, predicate)
        '''average_def_dist, max_def_dist = word_distance_to_property_definition(definition_properties[property_name], predicate,
                                                                              global_model)'''

        #print('---', average_def_dist, v['local_average'], v['global_average'], v['tf'])
        temp_list = [v['local_average'], v['global_average'], v['tf']]
        temp_score = (np.prod(temp_list)*len(temp_list))/sum(temp_list)

        try: temp_score = 1/-math.log(temp_score)
        except: temp_score = 0

        result_dict[k] = (temp_score, temp_score, v['tf'])
        normalized_values.append((temp_score, temp_score, v['tf']))
        normalized_labels.append(k)

    #{'local_average':local_average, 'local_max_dist': local_max_dist, 'global_average': global_average,
    #        'global_max_dist': global_max_dist, 'subject_dist': subject_dist, 'object_dist': object_dist}


    # normalize values
    normalized_values = MinMaxScaler().fit(normalized_values).transform(normalized_values)
    for k, v in zip(normalized_labels, normalized_values):
        result_dict[k] = v.tolist()

    #print('result_dict: ', result_dict)
    
    result_dict = dict(sorted(result_dict.items(), key = lambda v: v[1], reverse = True))
    return result_dict


def group_predicate(predicate_dict, top=10, show=False):

    group_dict = {}
    for k, v in predicate_dict.items():
        temp_list = k.split('-')
        key = temp_list[0] + '-' + '-'.join(temp_list[2:])
        key = key.strip('-')

        predicate = temp_list[1]
        temp_list = [*v]
        temp_list.insert(0, predicate)
        if (key not in group_dict):
            group_dict[key] = [temp_list]
        else:
            group_dict[key].append(temp_list)
        
    #group_dict = sorted(group_dict.items(), key = lambda v: (v[0]), reverse = True))

    if (show==False): return group_dict
        
    i = 1
    for k, v in group_dict.items():
        print('+', k)
        for x in v:
            if (i > top): break
            print('---', x)
            i = i + 1
        i = 1

    return group_dict

#...........................
#...........................

# get idf of a word
def create_idf_word(sentences, word):

    n = len(sentences)
    freq = 0 # raw frequency
    
    for s in sentences:
        freq += sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(word), s))
    
    return freq
        
# create idf dictionary
def create_idf_dict(sentences):

    n = len(sentences)
    result_dict = {}

    result_dict['%%SIZE%%'] = n # number of documents (sentences)
    
    for s in sentences:
        doc = nlp(s)
        for token in doc:
            if (str(token.text) not in result_dict):
                result_dict[str(token.text)] = create_idf_word(sentences, str(token.text))

    return result_dict
          
# get inverse document frequency, a sentence as a document
def get_idf(idf_dict, word):

    n = idf_dict['%%SIZE%%']
    freq = 0
    if (word in idf_dict):
        freq = idf_dict[word]

    # return -math.log((freq + 1)/n)
    return -math.log((freq+1)/n) + 1
    

# get frequency of a term in corpus, corpus as a document
def get_tf(counter, word):
    temp = (word,) # create key
    freq = 0
    freq = counter[temp] # raw frequency
    #n = len(counter)
    
    return math.log(freq+1) + 1

# count and rank the number of sentence by its redudant words
def rank_sentence_by_redundant_words(redundants):

    count_dict = {}
    for r in redundants:
        if (r not in count_dict):
            count_dict[r] = 1
        else:
            count_dict[r] += 1

    count_dict = sorted(count_dict.items(), key=lambda x: x[0])
    
    return count_dict

# show sentences by distance
def show_sentence_distance(labels, scores, redundants, sentences, redundant_word_list):
    i = 0
    for value in scores:
        print('#' + str(i) + ': ', value, labels[i], redundants[i], redundant_word_list[i], '---', sentences[i])
        i = i + 1

# show plot of sentences
def show_sentence_plot(labels, scores, redundants, sentences, redundant_word_list):

    #labels = []
    #scores = []
    #redundants = []
    #sentences = []
    #redundant_word_list = []

    #labels, scores, redundants, sentences, redundant_word_list = convert_corpus_to_distance(sen_list, best_sentences, model, counter)
    #labels, tokens = convert_corpus_to_vector1(word_corpus, best_sentences, model, counter)

    #tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    #new_values = tsne_model.fit_transform(tokens)


    plt.figure(figsize=(20, 20))
    for i in range(len(scores)):
        # s: size, color: color
        if (labels[i] == 'x'):
            plt.scatter(scores[i], redundants[i], s=20, color='blue') # marker = 's'
            plt.annotate('', xy=(scores[i], redundants[i]), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
        else:
            plt.scatter(scores[i], redundants[i], s=2)
            plt.annotate('', xy=(scores[i], redundants[i]), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

    plt.xlabel("Number of redundant words")
    plt.ylabel("Score")

    plt.title('Sentences in corpus')
    plt.savefig('show_sentence_plot.pdf')
    plt.show()
    

# show plot of predicates (Wikidata properties)
def show_predicate_plot(predicate_list, axis_labels):

    plt.figure(figsize=(12, 4))

    for index, predicate_dict in enumerate(predicate_list):
        labels = []
        values = []

        for k, v in predicate_dict.items():
            labels.append(k)
            values.append(v)

        #tsne_model = TSNE(perplexity=100, n_components=1, init='pca', n_iter=5000)
        #values = tsne_model.fit_transform(values)
        #values = decomposition(values, 'pca', dimension = 2)
        #values = MinMaxScaler().fit(values).transform(values)

        x = []
        y = []
        sizes = []
        i = 0
        for v in values:
            a = v[0]
            b = v[1]
            #print('+++', labels[i], a, b, int(v[2]*300)+1)
            x.append(a)
            y.append(b)
            sizes.append(int(v[2]*300)+1)
            i = i + 1
            
        plt.rcParams.update({'font.size':10})
        for i in range(len(x)):
            # s: size, color: color
            plt.scatter(1, 1, s=1, alpha=0.0)
            plt.scatter(index + 2, y[i], s=sizes[i], alpha=0.6) # marker = 's'
            if (i < 5):
                temp_label = labels[i][labels[i].index('-')+1:]
                #print('temp_label: ', temp_label)
                plt.annotate(temp_label, xy=(index + 2, y[i]), xytext=(2, 2), textcoords='offset points', ha='right', va='bottom',
                             alpha=0.9, fontsize=8)
            #fontsize=int(sizes[i]/10)+1

    plt.grid(color = 'grey', linestyle = 'dotted', linewidth = 1)
    #plt.gca().axes.get_xaxis().set_visible(False)

    # axis labels
    plt.xticks(range(2, len(axis_labels)+2), axis_labels)
    
    plt.show()

# get all qualifiers by Wikidata properties
def get_all_qualifiers(sen_df):
    result_list = []
    result_dict = {}

    for index, sen_row in sen_df.iterrow():
        temp_list = get_qualifier_items(sen_row)
        for t in temp_list:
            value = t[t.index(':') + 1:t.index('-')]
            result_list.append(value)

    result_list = list(set(result_list))
    result_list = sorted(result_list, key = lambda x: int(x[1:]))

    for r in result_list:
        root = get_wikidata_root(r)
        label = get_label(root)
        description = get_description(root)
        aliases = ' '.join(e for e in get_alias(root))

        def_string = label + ' ' + description + ' ' + aliases
        def_list = []
        doc = nlp(def_string)

        for token in doc:
            if (token.pos_ == "X"):
                continue
            if (token.pos_ == "PUNCT"):
                continue
            if (token.pos_ == "CCONJ"):
                continue
            if (token.pos_ == "ADP"):
                continue
            if (token.pos_ == "PRON"):
                continue
            if (token.pos_ == "PART"):
                continue
            if (token.pos_ == "DET"):
                continue
            if (token.dep_ == "punct"):
                continue
            def_list.append(token.text)

        def_list = list(set(def_list))

        #print('def_list:', r, def_list)
        result_dict[r] = def_list

    print('result_dict qualifiers: ', result_dict)
    return result_dict

# sentence plot by redundant words
def sentence_plot_by_redundant_words(total_cumulative_list, labels, plot_title, x_axis_label, y_axis_label):

    #cmap = plt.get_cmap('plasma')
    #colors = cmap(np.linspace(0, 1, len(labels)))

    colors = ['green', 'blue', 'red', 'coral', 'orchid', 'gray', 'gold']
    colorcyler = cycle(colors)

    lines = ['+', '*', '>', 'x', 'o', ':', '--']
    linecycler = cycle(lines)

    
    plt.rcParams.update({'font.size':10})
    
    plt.ylabel(y_axis_label)
    plt.xlabel(x_axis_label)
    #plt.title(plot_title)
    #plt.figure(figsize=(1,30))
    #plt.figure(figsize=(1, 1), dpi=1000)

    scale_factor = 30
    xmin, xmax = plt.xlim()
    plt.xlim(xmin * scale_factor, xmax * scale_factor)
    
    for cumulative_list, name, color in zip(total_cumulative_list, labels, colors):

        x, y = [], []
        i = 0
        for r in cumulative_list:
            x.append(r[0])
            y.append(r[1])
            i = i + 1

        plt.plot(x, y, next(linecycler), label=name, c=next(colorcyler))
        plt.legend()
        
        '''for i in range(len(y)):
            plt.scatter(x[i], y[i], s=2, color=color)'''

    
    #ymin, ymax = plt.ylim()

    
    #plt.ylim(ymin * scale_factor, ymax * scale_factor)

    plt.grid(color = 'grey', linestyle = 'dotted', linewidth = 0.5)
    
    plt.savefig('sentence_plot_by_redundant_words.pdf')
    plt.savefig('sentence_plot_by_redundant_words.svg')
    plt.show()
    plt.style.use('default') # reset style to default

# accumulative rate [0-1]
def cumulative_rate(rank_list):
    result_list = []
    total = sum([r[1] for r in rank_list])
    temp = 0
    for r in rank_list:
        temp += r[1]/total
        #print(temp)
        result_list.append([r[0], temp])
    
    #print(result_list)
    return result_list 

# minimums by redundant words
def minimums_by_redundant_words(scores, redundants):

    result_dict = {}
    for s, r in zip(scores, redundants):

        if (r not in result_dict):
            result_dict[r] = s
        else: # get min
            if s < result_dict[r]: result_dict[r] = s

    result_dict = sorted(result_dict.items(), key=lambda x: x[0])
    
    #print(result_dict)
    return result_dict

# linear regression
def linear_regression(x, y):

    print('x: ', x)
    print('y: ', y)
    x = np.array(x).reshape((-1, 1))
    y = np.array(y)

    model = ''
    try:
        model = LinearRegression().fit(x, y)
    except:
        model = LinearRegression().fit(x, y)
    
    r_sq = model.score(x, y)
    print('coefficient of determination:', r_sq)
    print('intercept:', model.intercept_)
    print('slope:', model.coef_)

    y_pred = model.predict(x)
    print('predicted response:', y_pred, sep='\n')
    
    mae = metrics.mean_absolute_error(y, y_pred)
    print('Mean Absolute Error:', mae)

    mse = metrics.mean_squared_error(y, y_pred)
    print('Mean Squared Error:', mse)

    rmse = np.sqrt(metrics.mean_squared_error(y, y_pred))
    print('Root Mean Squared Error:', rmse)

    result_dict = {}
    result_dict['y_pred'] = y_pred
    result_dict['intercept'] = model.intercept_
    result_dict['coef'] = model.coef_
    result_dict['r_sq'] = r_sq
    result_dict['mae'] = mae
    result_dict['mse'] = mse
    result_dict['rmse'] = rmse
    
    return result_dict

# linear regression plot
def linear_regression_plot(x, y, dict1, dict2, plot_title, x_axis_label, y_axis_label):

    plt.figure(figsize=(20, 20))
    for i, j in zip(x, y):
        
        plt.scatter(i, j, s=10, alpha=0.5)
        plt.annotate('', xy=(i, j), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

    axes1 = plt.gca()
    x_vals1 = np.array(axes1.get_xlim())
    y_vals1 = dict1['intercept'] + dict1['coef']*x_vals1
    print('x_vals1, y_vals1: ', x_vals1, y_vals1)
    plt.plot(x_vals1, y_vals1, '--')

    axes2 = plt.gca()
    x_vals2 = np.array(axes2.get_xlim())
    y_vals2 = dict2['intercept'] + dict2['coef']*x_vals2
    print('x_vals2, y_vals2: ', x_vals2, y_vals2)
    plt.plot(x_vals2, y_vals2)

    plt.grid(color = 'grey', linestyle = 'dotted', linewidth = 0.5)

    plt.savefig('linear_regression_plot.pdf')
    
    plt.show()

    #plt.style.use('default') # reset style to default


# filter noise by cumulative rate
def filter_noise_by_cumulative_rate(sentences, redundant_word_list, number_redundant_word_list, cumulative_list,
                                    rate = 0, top_words = 0):

    sentences_, redundant_word_list_, number_redundant_word_list_ = [], [], []
    
    if (rate == 0 and top_words == 0):
        return sentences, redundant_word_list, number_redundant_word_list

    bound = 0 # number of words used to filter
    # filter by rate only
    if (rate != 0 and top_words == 0):
        for c in cumulative_list:
            bound = c[0]
            if (c[1] > rate):
                break
    elif(rate == 0 and top_words != 0):
        bound = top_words

    if (bound == 0):
        return sentences, redundant_word_list, number_redundant_word_list
        
    for a, b, c in zip(sentences, redundant_word_list, number_redundant_word_list):
        if (c <= bound):
            sentences_.append(a)
            redundant_word_list_.append(b)
            number_redundant_word_list_.append(c)
            
    return sentences_, redundant_word_list_, number_redundant_word_list_

# filter noise by metrics
def filter_noise_by_metrics(df, field, frac=1, ascending=True):

    # convert to numeric
    df['local_tf_idf2'] = pd.to_numeric(df['local_tf_idf2'], errors='coerce') # standard
    df[field] = pd.to_numeric(df[field], errors='coerce') # metric
    df['length'] = pd.to_numeric(df['length'], errors='coerce') # number of redundant words

    # sort df
    sorted_df = df.sort_values(field, ascending=ascending)

    # get fraction
    df_len = len(sorted_df.index)
    n = int(df_len*frac)
    df_frac = sorted_df.head(n)
    
    # linear regression
    length_list = df_frac['length'].tolist()
    field_list = df_frac['local_tf_idf2'].tolist()
    #field_list = df_frac[field].tolist()
    linear_regression(length_list, field_list)

    
    '''for index, row in df_frac.iterrows():
        labeled_sentence_2 = row['labeled_sentence_2']

        length = row['length']
        label = row['label']
        score = row[field]

        print('--------------')
        print(label, length, score)
        print(labeled_sentence_2)'''
        
        

def filter_noise_by_clustering_method(df, field, compared_field, method, frac=1, ascending=True):

    """
        Not used ---
        # filter noise by DBSCAN (not use)
        # https://blog.dominodatalab.com/topology-and-density-based-clustering/
    """
    # convert to numeric
    df[compared_field] = pd.to_numeric(df[compared_field], errors='coerce') # standard
    df[field] = pd.to_numeric(df[field], errors='coerce') # metric
    df['length'] = pd.to_numeric(df['length'], errors='coerce') # number of redundant words

    # sort df
    sorted_df = df.sort_values(field, ascending=ascending)

    # get fraction
    df_len = len(sorted_df.index)
    n = int(df_len*frac)
    df_frac = sorted_df.head(n)

     # prepare data
    length_list = df_frac['length'].tolist()
    label_list = df_frac['label'].tolist()
    field_list_c = df_frac[compared_field].tolist()

    #field_list_c = df_frac[field].tolist()
    field_list = df_frac[field].tolist()

    # clustering
    new_list = []
    for x, y in zip(length_list, field_list):
        new_list.append([x, y])
    X = np.array(new_list)

    clustering = []
    if (method == 'dbscan'):
        clustering = DBSCAN(eps=3.25, min_samples=5).fit(X)
    elif (method == 'optics'):
        clustering = OPTICS().fit(X)
    elif (method == 'meanshift'):
        clustering = MeanShift().fit(X)
    elif (method == 'affinity'):
        clustering = AffinityPropagation().fit(X)
    elif (method == 'agglomerative'):
        clustering = AgglomerativeClustering(n_clusters=2, affinity = 'euclidean', linkage = 'complete').fit(X)
    elif (method == 'localoutliner'):
        # -1: outliner
        clustering =  LocalOutlierFactor(n_neighbors=100).fit_predict(X)
    else:
        print('Clustering method is not defined!!!')
        return
        
    create_2d_graph('x', 'y', X, label_list, clustering.labels_)
    
    # linear regression
    filtered_length_list, filtered_field_list = [], []
    for id, val in enumerate(clustering.labels_):
        if (val == 0): # first group (or most density group)
            filtered_length_list.append(length_list[id])
            filtered_field_list.append(field_list_c[id])
    linear_regression(filtered_length_list, filtered_field_list)

def create_2d_graph(x_axis_label, y_axis_label, values, labels, colors):
    plt.figure(figsize=(10, 7))
    plt.subplots_adjust(bottom=0.1)

    plt.scatter(values[:,0],values[:,1], c = colors, cmap='rainbow', s=6)

    for label, color, x, y in zip(labels, colors, values[:, 0], values[:, 1]):
        #str(color) + '-' + 'words' + str(label)
        if (label == 1):
            #plt.scatter(x, y, c = 'blue', s=6)
            plt.annotate(str(label), xy=(x, y), xytext=(2, 2), size=6, textcoords='offset points', ha='left', va='bottom')
        #else:
            #plt.scatter(x, y, s=6)

    colors = list(colors)
    counter = Counter(colors)
    print('Clusters: ', counter)
        
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)

    #plt.title('Topics by coherence (' + coherence + ') and itf')
    plt.grid(color = 'grey', linestyle = 'dotted', linewidth = 0.5)
    plt.show()

