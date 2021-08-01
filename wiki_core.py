#........................................................................................................
# Title: Wikidata claims (statements) to natural language (a part of Triple2Text/Ontology2Text task)
# Author: Ta, Hoang Thang
# Email: tahoangthang@gmail.com
# Lab: https://www.cic.ipn.mx
# Date: 12/2019
#........................................................................................................

import csv
import gc
import nltk
import os
import re
import requests
import spacy
import xml.etree.ElementTree as ET
import html

from collections import Counter
from datetime1 import *
from dateutil.easter import *
from dateutil.parser import *
from dateutil.relativedelta import *
from dateutil.rrule import *

from nltk import Tree
from nltk import ngrams
from nltk.tokenize.treebank import TreebankWordDetokenizer

from spacy import displacy
#from spacy.lang.en import English
#from spacy.tokenizer import Tokenizer
nlp = spacy.load('en_core_web_md')
nlp.add_pipe(nlp.create_pipe('sentencizer'), before='parser')

from read_write_file import *

boundary = re.compile('^[0-9]$')

tag_re = re.compile(r'<[^>]+>')
country_list = read_from_csv_file('corpus/demonyms.csv', ',', 'all') # country, nationality

# pre-define Wikidata's quantity properties and their units
def property_unit():
    property_unit_list = [['P1971', 'number of children', ['child', 'chidlren', 'son', 'sons', 'daughter', 'daughters']],
                          ['P1350', 'number of matches played/races/starts', ['match', 'matches', 'game', 'games', 'event', 'events', 'appearance', 'appearances']],
                          ['P1351', 'number of points/goals/set scored', ['goal', 'goals', 'point', 'points']],
                          ['P1129', 'national team caps', ['cap', 'caps', 'game', 'games', 'match', 'matches', 'event', 'events', 'appearance', 'appearances']],
                          ['P2097', 'term length of office', ['year', 'years', 'month', 'months', 'day', 'days']]]
    
    #property_unit.append(['P1087', 'Elo rating', []])
    #property_unit.append(['P1352', 'ranking', []])
    #property_unit.append(['P2415', 'personal best', ['second', 'seconds', 'minutes', 'minute', 'hour', 'hours']])
    #property_unit.append(['P1345', 'number of victims of killer', ['person', 'people', 'victim', 'victims', 'man', 'men', 'woman', 'women']])
    #property_unit.append(['P1355', 'number of wins', ['match', 'matches', 'game', 'games', 'event', 'events']])
    #property_unit.append(['P1356', 'number of losses', ['match', 'matches', 'game', 'games', 'event', 'events']])
    #property_unit.append(['P1357', 'number of draws/ties', ['match', 'matches', 'game', 'games', 'event', 'events']])
    #property_unit.append(['P1358', 'points for', ['point', 'points']])
    #property_unit.append(['P1359', 'number of points/goals conceded', ['match', 'matches', 'game', 'games', 'event', 'events']])
    
    return property_unit_list

# format text
def format_text(text):
    # remove html tags
    r = re.compile(r'<.*?>')
    text = re.sub(r, '', text)
    text = text.replace('“','"')
    text = text.replace('”','"')
    text = text.replace('’',"'")
    text = text.replace('‘',"'")
    text = text.replace('\n',' ')
    text = text.replace('\r',' ')
    text = text.replace('\n\r',' ')
    text = remove_emojis(text)
    text = text.strip()
        
    text = text.replace('&#39;',"'")
    text = text.replace('&quot;','"')     
    text = text.replace('&nbsp;',' ') # replace as space
    text = text.replace('&amp;','&')
        
    try: 
        text = html.unescape(text)
    except Exception as e:
        print('Error442: ', e)
    
    return text

# detokenizer -- it is not the most correct but helps
def detokenizer(token_list):
    return TreebankWordDetokenizer().detokenize(token_list)

# remove emojis, function is from Karim Omaya (stackoverflow.com)
def remove_emojis(data):
    emoj = re.compile('['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002500-\U00002BEF'  # chinese char
        u'\U00002702-\U000027B0'
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        u'\U0001f926-\U0001f937'
        u'\U00010000-\U0010ffff'
        u'\u2640-\u2642' 
        u'\u2600-\u2B55'
        u'\u200d'
        u'\u23cf'
        u'\u23e9'
        u'\u231a'
        u'\ufe0f'  # dingbats
        u'\u3030'
                      ']+', re.UNICODE)
    return re.sub(emoj, '', data)

# clear screen
def cls():
    os.system('cls' if os.name=='nt' else 'clear')

# print nltk tree of a sentence
def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_ # verbatim text content (unicode)

# get xml data by page title (English Wikipedia)
def get_xml_data_by_title(title):
    link = 'https://en.wikipedia.org/w/api.php?action=query&format=xml&rvprop=content&prop=extracts|revisions|pageprops|templates|categories&rvslots=main&titles=' + title
    response = requests.get(link)
    root = ET.fromstring(response.text)
    return root

# get XML's root of a wiki page by its wikidataID
def get_wikidata_root(wikidata_id):
    if (wikidata_id == None):
        wikidata_id = ''
    link = 'https://www.wikidata.org/w/api.php?action=wbgetentities&format=xml&ids=' + wikidata_id
    root = get_xml_data_by_url(link)
    if (root is None):
        return ''
    return root

# get property's datatype
def get_property_datatype(root):
    try:
        for x in root.find('./entities'):
            #print(x)
            value = remove_emojis(x.attrib['datatype'])
            if (value != ''):  
                return value
    except:
        pass
    return ''
    
# get xml data by url
def get_xml_data_by_url(link):
    response = requests.get(link)
    root = ET.fromstring(response.text)
    return root

# get English Wikipedia (enwiki) sitelink
def get_sitelink(root):
    try:
        for x in root.find('./entities/entity/sitelinks'):
            if (x.attrib['site'] == 'enwiki'):
                value = remove_emojis(x.attrib['title'])
                if (value != ''):  
                    return value
    except:
        pass
    return ''
    
# get wikidataID
def get_wikidata_id(root):
    for node in root.find('./query/pages/page'):
        if (node.tag == 'pageprops'):
            return node.attrib['wikibase_item']
    return ''

# get article content in HTML format
def html_content(root):
    text = ''
    for x in root.iter('extract'):
        text += x.text
    text = text.replace('\n',' ')    
    return text

# split content into sections, not use for h3, h4, h5
def get_content_by_section(text):

    secs = re.split('<h2>(.*?)</h2>', text)
    secs = [tag_re.sub(r' ', x) for x in secs] 
    secDict = dict()
    key = 'Beginning' # first part
    i = 0
    for x in secs:
        if (i%2 == 0):
            secDict[key] = x.strip() # text = format_text(text)
        else:
            key = x
        i = i + 1
    return secDict

# get text without sections
def get_text_not_by_section(text):
    #print('raw text: ', text)
    list_headers = re.findall('<(h1|h2|h3|h4|h5|h6)>(.*?)</(h1|h2|h3|h4|h5|h6)>', text)
    
    for x in list_headers:
        sub = ''
        try:
            sub += '<' + x[0] + '>' + x[1] + '</' + x[2] + '>'
            #print(sub)
            text = text.replace(sub, '')
        except:
            continue

    text = re.sub(tag_re, '', text)
    text = format_text(text)
    #print('text:', text)
    return text

# split sentences by comma
def sentence_list(text):
    text = text.replace(u'\xa0', u' ') # remove non-breaking space \xa0
    text = text.replace(u'"', u'') # remove "
    
    sents_list = text.split('.')    
    sents_list = [x.strip() for x in sents_list]
    sents_list = [x + '.' for x in sents_list]
    return sents_list

# split sentences by spaCy sentenizer
def sentence_list_sentencizer(text):
    text = text.replace(u'\xa0', u' ') # remove non-breaking space \xa0
    text = text.replace(u'"', u'') # remove "
 
    doc = nlp(text)

    sents_list = []
    for sent in doc.sents:
        sents_list.append(sent.text.strip())
           
    return sents_list

# get label (page name) of a Wiki page by its wikidataID
def get_label_by_wikidataID(wikidataID):
    link = 'https://www.wikidata.org/w/api.php?action=wbgetentities&format=xml&ids=' + wikidataID
    root = get_xml_data_by_url(link)           
    return get_label(root)

# get label (page name) by XML's root
def get_label(root):
    try:
        for x in root.find('./entities/entity/labels'):
            if (x.attrib['language'] == 'en-gb' or x.attrib['language'] == 'en'):
                value = remove_emojis(x.attrib['value'])
                if (value != ''):
                    return value
    except:
        pass
    return ''

# get a short description of a Wiki page by XML's root
def get_description(root):

    try:
        for x in root.find('./entities/entity/descriptions'):
            if (x.attrib['language'] == 'en-gb' or x.attrib['language'] == 'en'):
                value = remove_emojis(x.attrib['value'])
                if (value != ''):  
                    return value
    except:
        pass
    return ''       

# get values by property
def get_values_by_property(claims, property_name):
    result_list = []
    try:
        for c in claims:
            if (c[1][1] == property_name):
                k = c[1][3]
                root = get_wikidata_root(k)
                label = get_label(root)
                #wikidata_id = get_wikidata_id(root)
                result_list.append([k, label])
    except:
        pass

    return result_list
	
# get instance of (P31)
def get_instance_of(claims):
    result_list = []
    try:
        for c in claims:
            if (c[1][1] == 'P31'):
                k = c[1][3]
                root = get_wikidata_root(k)
                instance_name = get_label(root)
                #wikidata_id = get_wikidata_id(root)
                result_list.append([k, instance_name])
    except:
        pass

    return result_list

# get subclass of (P279)
def get_subclass_of(claims):
    result_list = []
    try:
        for c in claims:
            if (c[1][1] == 'P279'):
                k = c[1][3]
                root = get_wikidata_root(k)
                instance_name = get_label(root)
                #wikidata_id = get_wikidata_id(root)
                result_list.append([k, instance_name])
    except:
        pass

    return result_list
	
# get nationality (P27 - country of citizenship)
def get_nationality(claims):

    result_list = []
    try:
        for c in claims:
            if (c[1][1] == 'P27'):
                k = c[1][3]
                root = get_wikidata_root(k)
                country_name = get_label(root)
                result_list.append([k, country_name])
    except:
        pass

    return result_list
	
# get alias (other label names) of a Wiki page by XML's root
def get_alias(root):
    aliases = []
    try:
        for z in root.find('./entities/entity/aliases'):
            if (z.attrib['id'] == 'en-gb' or z.attrib['id'] == 'en'):
                for t in z:
                    value = remove_emojis(t.attrib['value'])
                    if (value != ''): 
                        aliases.append(value)
    except:
        pass
    return aliases
        
# get claims (Wikidata's statements) of a Wiki page --- single qualifier value
# drop object is not "wikibase-item" in WST-3 statements 
def get_claims(root, wikidataID):
  
    claim_list = [] # statement list
    s = wikidataID # s: subject (item identifier, wikidataID)
    p = ob = pt = pv = q = qt = qv = ''
    # p: predicate (property), ob: object (property value identifier)
    # pt: object type (property value type), pv: object value (property value)
    # q: qualifier, qt: qualifier type, qv: qualifier value

    for predicate in root.find('./entities/entity/claims'):        
        #print('************************')
        #print('Property: ', predicate.attrib['id'])
        p = remove_emojis(predicate.attrib['id']) # predicate
        for claim in predicate.iter('claim'):
            pt = remove_emojis(claim[0].attrib['datatype']) # property type
            #print('+', pt)            
            for obj in claim.find('mainsnak'):
                try:
                    try:
                        # obj.attrib['value'].encode('unicode-escape').decode('utf-8')
                        pv = remove_emojis(obj.attrib['value'])
                    except Exception as e:
                        #print('Error:', e)
                        pass
                    if (pv != ''):
                        continue
                    objdict = obj[0].attrib
                    if ('id' in objdict):
                        #print('--', objdict['id'])
                        ob = remove_emojis(objdict['id']) # qualifier
                    elif ('time' in objdict):
                        #print('--', objdict['time'])
                        pv = remove_emojis(objdict['time']) # time
                    elif ('amount' in objdict):
                        #print('--', objdict['amount'])
                        pv = remove_emojis(objdict['amount']) # amount
                    # capture other data types (globle coordinate, etc)
                    # ...
                    else:
                        print('--', 'empty')					
                except Exception as e:
                    #print('Error:', e)
                    pass
                    
            if (pv is not ''):
                r1 = [s, p, pv, pt] # change order different from get_claims2()
                #print('r1: ', r1, type(r1))
                claim_list.append(['r1', r1]) # WTS-1 statement
                pt = pv = '' # reset values
            elif (ob is not ''):
                r2 = [s, p, ob, pt] # change order different from get_claims2()
                try:
                    for x in claim.find('qualifiers'):
                        #print('----', x.attrib['id'], x.tag)
                        q = remove_emojis(x.attrib['id']) # qualifier identifier
                        qt = remove_emojis(x[0].attrib['datatype']) # qualifier data type
                        subr = [q, qt]
                            
                        for y in x.find('qualifiers'):
                            for z in y.iter('value'):
                                qv = '' # qualifier value
                                if ('id' in z.attrib):
                                    #print('--------', z.attrib['id'])
                                    qv = remove_emojis(z.attrib['id']) # value
                                elif ('time' in z.attrib):
                                    #print('--------', z.attrib['time'])
                                    qv = remove_emojis(z.attrib['time']) # value
                                elif ('amount' in z.attrib):
                                    #print('--------', z.attrib['amount'])
                                    qv = remove_emojis(z.attrib['amount']) # value
                                # capture other data types (globle coordinate, etc)
                                # ...
                                else:
                                    #print('--------', '')
                                    qv = '' # set to empty
                                if (qv is not ''): 
                                    subr.append(qv)
                                    r2.append(subr) # add a qualifier value
                                    qv = '' # set to empty for new iterator
                        
                except Exception as e:
                    #print('Error: ', e)
                    pass
                
                if (len(r2) > 4): 
                    claim_list.append(['r3', r2]) # WST-3 statement
                else:
                    #print('r2: ', r2, type(r2))
                    claim_list.append(['r2', r2]) # WST-2 statement

                ob = '' # reset value (important)
        #print('************************')

    '''for c in claim_list:
        print('-----------------')
        print(c)'''
        
    return claim_list                

# get claims (Wikidata's statements) of a Wiki page --- multiple qualifiers
def get_claims2(root, wikidataID):

    claim_list = [] # statement list
    s = wikidataID # s: subject (item identifier, wikidataID)
    p = ob = pt = pv = q = qt = qv = ''
    # p: predicate (property), ob: object (property value identifier)
    # pt: object type (property value type), pv: object value (property value)
    # q: qualifier, qt: qualifier type, qv: qualifier value

    if (root is None):
        return claim_list

    # loop each predicate (property)
    for predicate in root.find('./entities/entity/claims'):        
        #print('************************')
        #print('Property: ', predicate.attrib['id'])
        p = remove_emojis(predicate.attrib['id']) # predicate (property)
        for claim in predicate.iter('claim'):
            pt = remove_emojis(claim[0].attrib['datatype']) # property type
            #print('+', pt)
            for obj in claim.find('mainsnak'):
                try:
                    try:
                        # obj.attrib['value'].encode('unicode-escape').decode('utf-8')
                        pv = remove_emojis(obj.attrib['value'])
                    except Exception as e:
                        #print('Error:', e)
                        pass
                    if (pv != ''):
                        continue
                    objdict = obj[0].attrib

                    if ('id' in objdict):
                        #print('--', objdict['id'])
                        ob = remove_emojis(objdict['id']) # qualifier
                    elif ('time' in objdict):
                        #print('--', objdict['time'])
                        pv = remove_emojis(objdict['time']) # time
                    elif ('amount' in objdict):
                        #print('--', objdict['amount'])
                        pv = remove_emojis(objdict['amount']) # amount
                    # capture other data types (globle coordinate, etc)
                    # ...
                    else:
                        #print('--', 'empty')
                        pass
                except Exception as e:
                    #print('Error:', e)
                    pass
			
            # get qualifiers
            qual_properties = [t for t in claim.findall('qualifiers/property')]
            if (len(qual_properties) == 0):
                if (pt != 'wikibase-item'):
                    r1 = [s, p, pt, pv]
                    claim_list.append(['r1', r1]) # WST-1 statement
                else:
                    r2 = [s, p, pt, ob]
                    claim_list.append(['r2', r2]) # WST-2 statement
            else:
                if (pv != ''):
                    r3 = [s, p, pt, pv] # WST3-a
                else:
                    r3 = [s, p, pt, ob] # WST3-b 
                try:
                    for x in claim.find('qualifiers'):
                        #print('----', x.attrib['id'], x.tag)
                        q = remove_emojis(x.attrib['id']) # qualifier identifier
                        qt = remove_emojis(x[0].attrib['datatype']) # qualifier data type
                        subr = [q, qt]
                        children = x.getchildren()
                        for y in children:
                            for z in y.find('datavalue'):
                                qv = '' # qualifier value
                                if ('id' in z.attrib):
                                    #print('--------', z.attrib['id'])
                                    qv = remove_emojis(z.attrib['id']) # qualifier value
                                elif ('time' in z.attrib):
                                    #print('--------', z.attrib['time'])
                                    qv = remove_emojis(z.attrib['time']) # value
                                elif ('amount' in z.attrib):
                                    #print('--------', z.attrib['amount'])
                                    qv = remove_emojis(z.attrib['amount']) # value
                                # capture other data types (globle coordinate, etc)
                                # ...   
                                else:
                                    #print('--------', 'empty')
                                    qv = '' # set to empty
                                if (qv != ''):
                                    subr.append(qv)
                                    r3.append(subr) # add a qualifier value
                                    qv = '' # set to empty for new iterator
                except Exception as e:
                    #print('Error: ', e)
                    pass
                
                if (len(r3) > 4):
                    claim_list.append(['r3', r3]) # WST-3 statement
                else:
                    if (pt != 'wikibase-item'):
                        claim_list.append(['r1', r3]) # WST-1 statement
                    else:
                        claim_list.append(['r2', r3]) # WST-2 statement
            ob = pv = '' # reset values (important)
        #print('************************')    

    '''for c in claim_list:
        print('-----------------')
        print(c)'''

    return claim_list    

# get relations by type
def filter_claim_by_type(claim_list, type):
    result_list = []
    for x in claim_list:
        if (x[0] == type):
            result_list.append(x)
    return result_list        

# get dependency phrase
def dependency_phrase(node, results):
    if node.n_lefts > 0:
        for left in node.lefts:
            dependency_phrase(left, results)
            
    if node.n_rights > 0:
        #results.insert(0, node.orth_)
        results.append(node.orth_)
        for right in node.rights:
            dependency_phrase(right, results)
     
    else:
        results.append(node.orth_)
        return results

# ngram for phrase
def ngram_phrase(phrase):
    list_result = []
    n = len(phrase)
    for i in range(0, n):
        list_result += ngrams(phrase.split(), i)
    return sorted(list(set(list_result)))

#get dependency nouns, noun phrases
def get_dependency_nouns(sentence, word, word_root, exclude_list):

    doc = nlp(sentence)

    #named entities
    ent_list = []
    for ent in doc.ents:
        ent_list.append([ent.text, ent.label_])
    #print('ent_list: ', ent_list)

    #noun chunks
    '''for chunk in doc.noun_chunks:
        print(chunk.text)'''

    # get custom stop words
    custom_stop_words = ['—', '-', '/', '|', ':']
    for token in doc:
        #print(token.text, token.pos_, token.dep_)
        if (token.pos_ != 'NOUN' and token.pos_ != 'PROPN' and token.pos_ != 'NUM' and token.text not in [x[0] for x in ent_list]):
            custom_stop_words.append(token.text)
    custom_stop_words = list(set(custom_stop_words))
    #print('custom_stop_words: ', custom_stop_words)

    # get dependency raw strings
    raw_list = []
    start_pos = -1
    for token in doc:
        if (str(token.head) == word_root):
            start_pos = token.i
            break 
    for token in doc:
        if (str(token.head) == word_root):
            temp = []
            dependency_phrase(token, temp)
            raw_list.append(temp)
    
    #print('raw_list: ', raw_list)

    # get dependency nouns/noun phrases
    ngram_list = []
    for x in raw_list:
        temp = ' '.join(str(e.strip()) for e in x)
        #print('temp, word:---', temp, '---',word)
        if (temp not in word and word not in temp):
            ngram_list.append(ngram_phrase(temp))

    filter_list = []
    for x in ngram_list:
        for y in x:   
            if (len(y) == 1):
                temp = str(y[0]).strip()
                if (temp not in custom_stop_words):
                    filter_list.append(temp)
            else:
                whole = ' '.join(str(e.strip()) for e in y)
                head = str(y[0]).strip()
                last = str(y[len(y)-1]).strip()
                if (head not in custom_stop_words and last not in custom_stop_words):
                    filter_list.append(whole)

    filter_list = sorted(list(set(filter_list)))
    final_list = []

    for f in filter_list:
        flag = True
        for e in exclude_list:
            temp1 = e.split()
            temp2 = f.split()
            temp = list(set(temp1) & set(temp2))
            if (len(temp) > 0):
                flag = False
                
        if (flag == True):
            final_list.append(f)

    final_list = sorted(final_list, key = len, reverse = True) # sort by length of item
    #print('final_list: ', final_list)
    
    return_list = []
    for item in final_list:
        head = -1
        tail = -1
        word_type = 'NOUN CHUNK'
        temp_item = item.split()
        #print('temp_item: ', temp_item)
        if (len(temp_item) > 1):
            #head, tail = find_token_index_for_term(sentence, item)
            #return_list.append([item, head, tail, word_type])

            temp_list = find_all_token_index_for_term(sentence, item)
            for t in temp_list:
                temp_item = [item, t[0], t[1], word_type]
                if (len(return_list) == 0):
                    return_list.append(temp_item)
                else:
                    head, tail = find_term_position_by_list(item, return_list)
                    if (head != -1 and tail != -1 and t[0] >= head and t[1] <= tail):
                        return_list.append(temp_item)
                    elif (t[0] >= head and t[1] <= tail):
                        return_list.append(temp_item)
                               
            #print('return_list 1: ', return_list)
            
        else:
            #head = find_token_index(sentence, item)
            #return_list.append([item, head, head, word_type])
            temp_list = find_all_token_index(sentence, item)

            for t in temp_list:
                temp_item = [item, t, t, word_type]
                if (len(return_list) == 0):
                    return_list.append(temp_item)
                else:
                    head, tail = find_term_position_by_list(item, return_list)
                    if (head != -1 and tail != -1 and t >= head and t <= tail):
                        return_list.append(temp_item)
            
            #print('return_list 2: ', return_list)
                   
    '''try:
        [to_nltk_tree2(sent.root).pretty_print() for sent in doc.sents]
    except:
        print('Tree can not be displayed!')
    print('----------------------------------------------------')'''

    #print('return_list: ', return_list)
    return return_list


def find_term_position_by_list(term, result_list):
    for item in result_list:
        if (term in item[0]):
            return item[1], item[2]
    return -1, -1
    
def check_sentence(s):
    print('')
    print('')
    print('')
    print('*********************************NEW SENTENCE*********************************')
    print(s)
        
    doc = nlp(s)
    #for token in doc:
        ###print('token: ', token.text, token.dep_, token.pos_)

    root = get_root(s)
    
    try:
        [to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]
    except:
        print('Tree can not be displayed!')

    subject = get_subject(s, root)
    print('subj: ', subject)
    
    subjects = get_all_subjects(s, root)
    print('subjs: ', subjects)
    
    print('main verb: ', root)
    
    pverbs = get_root_with_preps(s)
    print('prep verb: ', pverbs)
    
    verbs = get_verb(s)
    print('verb: ', verbs)
       
    # property types can be height, weight, dimension, etc (not only dates)
    dates = get_date(s)
    print('date: ', dates)
    
    entities = get_entities(s)
    print('entity list: ', entities)
    
    #print('-----------------------------------')
    return subject, subjects, root, pverbs, verbs, dates, entities

# include clauses: that, which, where, etc
# map a sentence to a r3 statement 
def analyze_single_sentence(s, page_name, wikidataID, claim_list, subject, subject_name_list, entities, object_name_list, dates, claims): 

    match_subject_list = match_subject(wikidataID, subject, subject_name_list)
    #print('Match Subject: ', match_subject_list)

    match_object_list = match_object(s, wikidataID, entities, object_name_list)
    #print('Match Object: ', match_object_list)

    # add datetimes to property list to maximize the matching ability
    property_list = [e for e in entities]
    for d in dates:
        flag = False
        for p in property_list:
            if (d[0] == p[0] and d[1] == p[1] and d[2] == p[2]):
                flag = True
                break
        if (flag != True):
            property_list.append(d)
    #print('property_list: ', property_list)    
    
    match_prop1, match_prop2, match_prop3 = match_property(s, wikidataID, match_object_list, property_list, claims)
    #print('Match Property by DateTime: ', match_prop1)
    #print('Match Property by Quantitative: ', match_prop2)
    #print('Match Property by String: ', match_prop3)

    match_property_list = []
    for x in match_prop1:
        match_property_list.append(x)
    for y in match_prop2:
        match_property_list.append(y)
    for z in match_prop3:
        match_property_list.append(z)

    print('match_property_list: ', match_property_list)
    return match_subject_list, match_object_list, match_property_list

# we consider that a single sentence has only 1 subject
def check_single_sentence(subjects):
    if (len(subjects) > 1):
        count = 0
        for x in subjects:
            if (x[3] == 'PRON' or x[3] == 'PROPN'):
                count += 1
        if (count >= 2):    
            return False
    return True

# single sentence, clause sentence, etc
def analyze_sentence(sentences, page_name, wikidataID, claim_list, file_name):

    subject_name_list, poss = get_list_human_name(page_name, claim_list)
    #print('subject_name_list: ', subject_name_list)

    r3 = filter_claim_by_type(claim_list, 'r3') #get r3 relations
    #print('r3: ', r3)
    
    object_name_list = get_object_name_list(r3)

    for s in sentences:
        try:
            subject, subjects, root, pverbs, verbs, dates, entities = check_sentence(s)
            if (check_single_sentence(subjects) == True):     
                match_subj, match_obj, match_prop = analyze_single_sentence(s, page_name, wikidataID, claim_list, subject,
                                                                   subject_name_list, entities, object_name_list, dates, r3)             
                write_csv(match_subj, match_obj, match_prop, root, pverbs, verbs, wikidataID, s, file_name, poss)
                
        except Exception as e:
            print('Error: ', e) 
            continue
    return 

# find index by item as a sublist
def find_index_sublist_list(find_list, keyword):
    for i, item in enumerate(find_list):
        #print(i, item)
        if (keyword in item): # find first item
            return i
    return -1 # not found  
 
# write a single line to *.csv file
def write_csv(match_subj, match_obj, match_prop, root, pverbs, verbs, wikidataID, s, file_name, poss):

    try:
        if (len(match_subj) > 0 and len(match_obj) > 0 and len(match_prop) > 0):
            print('')
            print('...........................CREATE DATA FOR SAVING TO CORPUS...........................')

            # Type
            col1 = 'S1'
            print('col1: ', col1)

            print("match_prop:------", match_prop)

            predicate_list = []
            object_list = []
            property_list = []
            
            for p in match_prop:
                predicate_list.append(str(p[1][1]))

            for p in match_prop:
                object_list.append(str(p[1][2]))

            for p in match_prop:
                property_list.append(str(p[1][3]))

            predicate_list = set(predicate_list)
            object_list = set(object_list)
            property_list = set(property_list)

            # Relation
            col2 = wikidataID + ', ' + '-'.join(str(x) for x in predicate_list) + ', ' + '-'.join(str(x) for x in object_list) + ', ' + '-'.join(str(x) for x in property_list)
            col21 = wikidataID
            col22 = '-'.join(str(x) for x in predicate_list)
            col23 = '-'.join(str(x) for x in object_list)
            col24 = '-'.join(str(x) for x in property_list)
            print('col2: ', col2)

            # Raw sentence#
            col3 = s
            print('col3: ', s)

            # Subject
            col4 =  str(match_subj[0]).strip('[]')
            print('col4: ', col4)

            # Subject matching
            col5 =  str(match_subj[1]).strip('[]')
            print('col5: ', col5)

            # Root
            col6 =  str(root).strip('[]')
            print('col6: ', col6)

            # Prepositional verb
            col7 =  str(pverbs).strip('[]')
            print('col7: ', col7)

            # Verbs
            col8 = ''
            for x in verbs:
                col8 +=  str(x).strip('[]') + '|'

            if (len(col8) > 1):
                col8 = col8[0:len(col8)-1]
            print('col8: ', col8)

            # Predicate matching 
            col9 =  ''
            temp_list = []
            for x in match_obj:
                if (x[1][0] not in temp_list):
                    temp_list.append(x[1][0])
            for y in temp_list:
                col9 += str(y).strip('[]') + '|'

            if (len(col9) > 1):
                col9 = col9[0:len(col9)-1]    
            print('col9: ', col9)

            # Object 
            col10 =  ''
            for x in match_obj:
                col10 += str(x[0]).strip('[]') + '|'
            if (len(col10) > 1):
                col10 = col10[0:len(col10)-1]    
            print('col10: ', col10)

            # Object matching
            col11 =  ''
            for x in match_obj:
                col11 += str(x[1][1]).strip('[]') + '|'
            if (len(col11) > 1):
                col11 = col11[0:len(col11)-1]    
            print('col11: ', col11)

            # Properties 
            col12 =  ''
            list_temp1 = []
            for x in match_prop:
                col12 += str(x[0]).strip('[]') + '|'
                list_temp1.append(x[0][0])
            if (len(col12) > 1):
                col12 = col12[0:len(col12)-1]    
            print('col12: ', col12)

            # Check property repetition
            listTemp2 =  list_temp1[:]

            ###print('list_temp1 -- list_temp1: ', list_temp1, list_temp1)
            if (len(list_temp1) != len(list(set(listTemp2)))):
                return ''

            # Property matching
            col13 =  ''
            for x in match_prop:
                col13 += str(x[1]).strip('[]') + '|'
            if (len(col13) > 1):
                col13 = col13[0:len(col13)-1]    
            print('col13: ', col13)

            # Labeled sentence 1
            col14 = s
            doc = nlp(s)
            list1 = []
            for token in doc:
                list1.append(token.text)

            #subject indexes
            sindex1 = match_subj[0][1]
            eindex1 = match_subj[0][2]
            for i, val in enumerate(list1):
                if (i in range(sindex1, eindex1 + 1)):
                    list1[i] = '[s]'

            #object indexes
            sindex2 = []
            eindex2 = []
            obj_name_list = []
            #obj_id_list = []

            print('match_obj:' , match_obj)
            for x in match_obj:             
                obj_name_list.append(x[1][1][0:])
                sindex2.append(x[0][1])
                eindex2.append(x[0][2])
            print('obj_name_list: ', obj_name_list)    
            #print(sindex2, eindex2)
                
            list11 = list1[:] # create new list
            for i, val in enumerate(list1):
                #print('###', i, val)
                for j, val1 in enumerate(sindex2):
                    #print('###___', j, val1)
                    if (i in range(sindex2[j], eindex2[j] + 1)):
                        #k = ' '.join(list1[sindex2[j]:eindex2[j] + 1]) # error  [o-1]
                        k = detokenizer(list1[sindex2[j]:eindex2[j] + 1])

                        l = find_index_sublist_list(obj_name_list, k)
                        #print('l: ', l)

                        if (l == -1): # in case can not find the position of object
                            return
                        
                        list11[i] = '[o' +  str(l) + ']' # [o-1] ???
                        #print('list11[i]: ', list11[i])         
            list1 = list11

            print('match_prop: ', match_prop)
            # property indexes
            sindex3 = [] # start indexes
            eindex3 = [] # end indexes
            plabel3 = [] # property labels
            oid3 = [] # object id
                                                                                                                                                                                                          
            for x in match_prop:
                sindex3.append(x[0][1])
                eindex3.append(x[0][2])
                plabel3.append(x[1][3])
                oid3.append(x[1][2])

            # can not identify if there are more than 2 qualifiers mapping to the same text (conflict case)
            for i, val in enumerate(list1):
                for j, val1 in enumerate(sindex3):
                    #print(i, j, sindex3[j])
                    if (i in range(sindex3[j], eindex3[j] + 1)):
                        try:
                            l = find_index_sublist_list(obj_name_list, oid3[j])
                            
                            list1[i] = '[o' + str(l) + ':' +  plabel3[j] + '-qualifier]'

                            print('l, list1[i]: ', l, list1[i])
                        except Exception as e:
                            print('Error: ', e) 
                        
                        # print('-----: ', l, oid3[j])

            for i, val in enumerate(list1):
                #print('i, val: ', i, val)
                if (val == 'a' or val == 'an'):
                    list1[i] = '[det:a-an]'
                if (val == 'the'):
                    list1[i] = '[det:the]'
                if(val == poss):
                    #print('poss: ', poss)    
                    list1[i] = '[s:poss]'

            #print('list1[i]:', list1)     
            col14 = ' '.join(str(e).strip() for e in list1 if e != '')            
            print('col14: ', col14)

            # -----------------------------------------------------------------
            # map object's statements (extra matching)
            match_obj_names = []
            entities = []
            result_list = []
            
            print('match_obj: +++++++++', match_obj)

            exclude_list = []
            props = [x[0][0] for x in match_prop]
            objects = [x[0][0] for x in match_obj]
            exclude_list = props + objects
            exclude_list.append(match_subj[0][0])
            print('exclude_list: ', exclude_list)
            
            if (len(match_obj) > 0):
                for o in match_obj:
                    print('object: ', o)

                    if (len(match_obj) == 1): # only 1 object (maximize mapping abilities)
                        entities = get_dependency_nouns(col3, root[0], root[0], exclude_list) # start from root
                    else:
                        if (len(o[2]) == 0): # Ez
                            entities = get_dependency_nouns(col3, root[0], root[0], exclude_list) # start from root
                        else:
                            entities = o[2]
                    
                    print('entities: ', entities)
                    
                    map_r1 = [] #not so popular usage
                    map_r2 = []
                    try:
                        map_r1, map_r2 = match_object_claims(wikidataID, o[1][1][0], o[1][1][1], entities)
                    except Exception as e:
                        print('Error: ', e)
                        
                    if (len(map_r2) > 0):
                        result_list.append(map_r2)
                        
            print('result_list:', result_list)

            replace_list = [] 
            for relation in result_list:
                for item in relation:
                    objectid_temp = item[1]
                    object_name = item[2]
                    wikidataID_temp = item[3][1][1]
                    label_temp = get_label_by_wikidataID(wikidataID_temp)
                    replace_list.append([objectid_temp, object_name, wikidataID_temp, label_temp, item[4]])

            print('replace_list: ', replace_list) 

            # evaluate the replace list, prioritize larger terms
            replace_list = evaluate_replace_list(replace_list)

            for rep in replace_list:
                for i, val in enumerate(list1):
                    if (i in range(rep[4][1], rep[4][2] + 1)):
                        l = find_index_sublist_list(obj_name_list, rep[1])
                        list1[i] = '[o' + str(l) + ':' + str(rep[2]) + '-' + str(rep[3].replace(' ','_')) + ']'
            
            # -----------------------------------------------------------------
            # Labeled sentence 2  --- use for forming patterns
            print('list1: ', list1)
            list2 = []
            for i, val in enumerate(list1):
                try:
                    if (list1[i+1] == val and '[' in val and ']' in val):
                        continue
                    else:
                        list2.append(list1[i])
                except Exception as e:
                    print('Error: ', e)
                    if (i == len(list1) - 1):
                        list2.append(list1[i])
                    continue
            print('list2: ', list2)
            col15 = ' '.join(str(e) for e in list2)
            print('col15: ', col15)

            #list order of statements --- Order 1
            list3 = []

            matches = []
            matches = re.findall(r'[[\S]*]', col15)
            for m in matches:
                #print('m ne:', m)
                list3.append(m)

            # Order 1
            col16 = ','.join(str(e) for e in list3)
            print('col16: ', col16)

            #list order of statements --- Order 2
            list4 = col16.split(',')
            #print('list4: ', list4)
            list5 = []
            for val in list4:
                if (val not in list5):
                    list5.append(val)

            # Order 2        
            col17 = ','.join(str(e) for e in list5)
            print('col17: ', col17)

            #print('file_name: ....', file_name)
            with open(file_name, 'a', newline='', encoding='utf-8') as csvfile:
                fw = csv.writer(csvfile, delimiter='#', quoting=csv.QUOTE_MINIMAL)
                fw.writerow([col1, col21, col22, col23, col24, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12,
                             col13, col14, col15, col16, col17])
            csvfile.close()

            print('...........................WRITE SUCCESSFULLY TO CORPUS...........................')
    except Exception as e:
        print('Error: ', e)
        return

# evaluate mapping list for relations of objects, properties, etc
def evaluate_replace_list(replace_list):

    # [['Q242924', 'Martine McCutcheon', 'P106', 'occupation', ['singer', 6, 6, 'NOUN CHUNK']]]
    temp_list = [r[4][0] for r in replace_list]

    remove_list = []

    for t in temp_list:
        for r in replace_list:
            if (t != r[4][0] and t in r[4][0]):
                remove_list.append(r)
                break

    for r in remove_list: 
        try: 
            replace_list.remove(r) 
        except ValueError: 
            pass
                
    return replace_list
    
# only numbers NOT the writing representation of numbers (ex: nighteen eighty four -- future task)
def get_numbers(s, dates):
    return

# get entities in a sentence (NER, noun chunks)    
def get_entities(s):
    doc = nlp(s)
    
    list_ent_raw = []
    result_list = []

    for ent in doc.ents:
        temp = ent.text.split()
        if (len(temp) == 1):
            result_list.append([ent.text, ent.start, ent.end-1, ent.label_, ent.text])
        elif(len(temp) > 1):
            result_list.append([ent.text, ent.start, ent.end-1, ent.label_, temp[-1]])
        
        list_ent_raw.append(ent.text)

    list_ent_raw = list(set(list_ent_raw))
    #print('list_ent_raw: ', list_ent_raw)
    
    for chunk in doc.noun_chunks:
        try:
            if chunk.text not in list_ent_raw:
                head = -1
                tail = -1  

                #print('chunk.text: ', chunk.text) 
                temp = [token.text for token in nlp(chunk.text)]        
                #print('temp ne getenities: ', temp)
        
                if (len(temp)>1):
                    head, tail = find_token_index_for_term(s, chunk.text)
                    result_list.append([chunk.text, head, tail, 'NOUN CHUNK', chunk.root.text])
                else:
                    head = find_token_index(s, temp[0])
                    result_list.append([chunk.text, head, head, 'NOUN CHUNK', chunk.root.text])
        except:
            pass
        
    return result_list

# get entities in a sentence (NER, noun chunks, chunk root, etc)
def get_entities2(s):   
    matches = re.findall(r'[[\S]*]', s)
    for m in matches:
        s = s.replace(m, '')

    #print('s hai te: ', s)    
    doc = nlp(s)  
    list_ent_raw = []
    result_list = []

    for ent in doc.ents:
        result_list.append([ent.text, ent.start, ent.end-1, ent.label_])
        list_ent_raw.append(ent.text)

    list_ent_raw = list(set(list_ent_raw))

    #print('list_ent_raw hai te:', list_ent_raw)
    for chunk in doc.noun_chunks:

        try:
            if chunk.text not in list_ent_raw:

                head = -1
                tail = -1
            
                temp = [token.text for token in nlp(chunk.text)]        
       
                if (len(temp)>1):
                    head, tail = find_token_index_for_term(s, chunk.text)
                    result_list.append([chunk.text, head, tail, 'NOUN CHUNK'])
                else:
                    head = find_token_index(s, temp[0])
                    result_list.append([chunk.text, head, head, 'NOUN CHUNK'])

            if chunk.root.text not in list_ent_raw:
                head = -1
                tail = -1  

                temp = [token.text for token in nlp(chunk.root.text)]        

                if (len(temp)>1):
                    head, tail = find_token_index_for_term(s, chunk.root.text)
                    result_list.append([chunk.root.text, head, tail, 'NOUN CHUNK'])
                else:
                    head = find_token_index(s, temp[0])
                    result_list.append([chunk.root.text, head, head, 'NOUN CHUNK'])

            if (chunk.text == chunk.root.text):
                continue

            word_before_root = chunk.text
            word_before_root = word_before_root.replace(chunk.root.text, '').strip()

            if word_before_root not in list_ent_raw:
                head = -1
                tail = -1  

                temp = [token.text for token in nlp(word_before_root)]        
                if (len(temp) > 1):
                    head, tail = find_token_index_for_term(s, word_before_root)
                    result_list.append([word_before_root, head, tail, 'NOUN CHUNK'])
                else:
                    head = find_token_index(s, temp[0])
                    result_list.append([word_before_root, head, head, 'NOUN CHUNK'])
        except:
            pass

    #print('result_list: ', result_list)
    return result_list
    
# find head, tail index for a term or phrase (NOT a single word) --- first match search
def find_token_index_for_term(s, term):

    list1 = [token.text for token in nlp(term)]
    list2 = [token.text for token in nlp(s)]
    headw = list1[0]

    head = -1
    tail = -1

    check = True
    for i, val in enumerate(list2):
        check = True
        if (val == headw):
            head = i
            #print('loop: ',i + 1, i + len(list1))
            for j in range(i + 1, i + len(list1)):
                #print('-----#') 
                if (list2[j] != list1[j-i]):    
                    check = False
                    head = -1
                    tail = -1
                    break
                else:
                    tail = j
            if (head != -1 and tail != -1):
                return head, tail
        else:
            continue
            
    return head, tail

def find_all_token_index_for_term(s, term):

    list1 = [token.text for token in nlp(term)]
    list2 = [token.text for token in nlp(s)]
    headw = list1[0]
    result_list = []

    head = -1
    tail = -1

    check = True
    for i, val in enumerate(list2):
        check = True
        if (val == headw):
            head = i
            #print('loop: ',i + 1, i + len(list1))
            for j in range(i + 1, i + len(list1)):
                #print('-----#') 
                if (list2[j] != list1[j-i]):    
                    check = False
                    head = -1
                    tail = -1
                    break
                else:
                    tail = j
            if (head != -1 and tail != -1):
                result_list.append([head, tail])
        else:
            continue
            
    return result_list
    
# find token position of a word in sentence --- first match search
def find_token_index(s, word):
    doc = nlp(s)
    index = -1 # not found
    for token in doc:
        if (token.text == word):
            return token.i
    return index

# find all token positions of a word in sentence
def find_all_token_index(s, word):
    doc = nlp(s)
    result_list = []
    
    for token in doc:
        if (token.text == word):
            result_list.append(token.i)

    return result_list
    
# position of a date term
def get_date(s):
    doc = nlp(s)
    result_list = []
    text = ''
    for ent in doc.ents:
        if (ent.label_ == 'DATE'):

            head = -1
            tail = -1
            
            #print('date.text: ', ent.text)
            temp = [token.text for token in nlp(ent.text)]
            #print('temp ne get_date:', temp)

            if (len(temp)>1):
                head, tail = find_token_index_for_term(s, ent.text)
                result_list.append([ent.text, head, tail, 'DATE'])
            else:
                head = find_token_index(s, temp[0])
                result_list.append([ent.text, head, head, 'DATE'])
                
    return result_list

# get root of a sentence
def get_root(s):
    doc = nlp(s)
    root_index = -1
    root = ''
    for token in doc:
        if (token.dep_ == 'ROOT' and token.pos_ == 'VERB'):
            root_index = token.i
            root = token.text
            #print('###', root, root_index)
            break
    return [root, root_index]

#find prepositional verbs (transfer to, move to, go by, etc)
def get_root_with_preps(s):
    doc = nlp(s)
    root_index = -1
    root = ''
    temp_index = -1
    for token in doc:
        if (token.dep_ == 'ROOT' and token.pos_=='VERB'):
            root_index = token.i
            root = token.text
            temp_index = token.i + 1
            while(doc[temp_index].dep_=='prep'):
                root = root + ' ' + doc[temp_index].text
                temp_index += 1

            if (len(root.split()) < 2): #if can not find any preps after verb
                temp_index = root_index
            #print('---',(root))
            break
        
    return [root, root_index, temp_index]  

# get all verbs (single words)
def get_verb(s):
    doc = nlp(s)
    result_list = []
    index = -1
    label = ''
    for token in doc:
        if (token.pos_=='VERB' and (token.dep_=='relcl' or token.dep_=='ROOT'
                                    or token.dep_=='auxpass' or token.dep_=='conj'
                                    or token.dep_=='advcl' or token.dep_=='aux' or token.dep_=='ccomp')):
            index = token.i
            label = token.text
            result_list.append([label, index])
    return result_list        

# get all subjects
def get_all_subjects(s, root):
    doc = nlp(s)
    list_subj = []
    list_subj2 = []
    if(root[1] == 0):
        return ''

    head = -1
    tail = -1
    #print(root)
    for token in doc:
        if (token.dep_ == 'nsubj' or token.dep_ == 'nsubjpass'):
            subj = str(token.text)
            head = token.i
            list_subj.append([subj, head, token.pos_])

    for np in doc.noun_chunks:
        for x in list_subj:
            if (x[0] in np.text):
                x[0] = np.text
               
    for ent in doc.ents:
        for x in list_subj:
            if (x[0] in ent.text):
                x[0] = ent.text

    for x in list_subj:
        head = -1
        tail = -1
        
        temp = [token.text for token in nlp(x[0])]
             
        #print('temp ne get_all_subjects:', temp)
        
        if (len(temp)>1):
            head, tail = find_token_index_for_term(s, x[0])
            list_subj2.append([x[0], head, tail, x[2]])
        else:
            head = find_token_index(s, temp[0])
            list_subj2.append([x[0], head, head, x[2]])
                 
    return list_subj2
  
# get a main subject 
def get_subject(s, root):
    doc = nlp(s)
    subj = ''
    if(root[1] == 0):
        return ''
    
    head = -1
    tail = -1
    ne = '' # named entity
    ###print(root)
    for token in doc:
        if ((token.dep_ == 'nsubj' or token.dep_ == 'nsubjpass') and token.i < root[1]):
            subj = str(token.text)
            head = token.i
            break
    for np in doc.noun_chunks:
        temp1 = np.text.split()
        if (subj in temp1):
            subj = np.text
            ne = 'NOUN CHUNK'
            break
    for ent in doc.ents:
        temp2 = ent.text.split()
        if (subj in temp2):
            subj = ent.text
            ne = ent.label_
            break

    #print('subj: ', subj)
    temp = [token.text for token in nlp(subj)]        
    #print('temp ne get_subject:', temp)

    if (len(temp)>1):
        head, tail = find_token_index_for_term(s, subj)
        
    else:
        #print('temp:', temp, s)
        head = find_token_index(s, temp[0])
        tail = head

    return [subj, head, tail, ne]

# get list of properties by objectID
def get_property_by_object(objectID, claims):
    property_list = []
    for x in claims:
        #print('x[1][2]', x[1][2], objectID)
        if (x[1][2] == objectID):
            for y in range(4, len(x[1])):
                property_list.append(x[1][y])
    #print('property_list:', property_list)        
    return property_list   

# convert datetime to type 'ISO 8601'
def convert_datetime_ISO8601(datetime_label):
    try:
        dt = datetime.strptime(datetime_label, '%Y-%m-%dT%H:%M:%SZ')
        return dt
    except Exception as e:
        
        try:
            temp = datetime_label.split('T')
            temp1 = temp[0].split('-')
            if (temp1[1] == '00'):
                temp1[1] = '01'
            if (temp1[2] == '00'):
                temp1[2] = '01'
            label = '-'.join(e for e in temp1)
            label = label + 'T' + temp[1]  
            dt = datetime.strptime(label, '%Y-%m-%dT%H:%M:%SZ')
            return dt       
        except Exception as e:
            pass
    return False

# compare a datetime (year) with a matched list
def compare_datetime(dt, claims):
    try:
        dt1 = parse(dt)
    except:
        return False
    for x in claims:
        #print('--', x, dt1)
        if (x[1] == dt1):
            return [x[0], x[1].strftime('%Y-%m-%dT%H:%M:%SZ')]
        elif (x[1].year == dt1.year):
            return [x[0], x[1].strftime('%Y-%m-%dT%H:%M:%SZ')]
    return False

# format datetime string before converting
# ex: summer 2004 => 2004, 2004 => 2004-1-1
def format_date_string(dt):
    if (len(dt) < 5 and dt.isdigit()): # year: 2003, 200, 20, 2
            #print('--', dt + '-01-01')
            return dt + '-01-01' # 2004 => 2004-01-01
    else:
        try:
            x = parse(dt)
            #print('---', dt)
            return dt
        except:
            temp = [int(x) for x in dt.split() if x.isdigit()]
            if (len(temp) == 1):
                y = temp[0]
                #print(str(y) + '-01-01')
                return str(y) + '-01-01'
    return dt    

# match datetime 
def match_date_time(sentence, wikidataID, list_object, list_property, claims):
    list1 = [] #wikidata properties
    list2 = [] #convert datetimes
    list3 = [] #results

    '''print('list_object:_____', list_object)
    print('list_property:_____', list_property)
    print('claims:_____', claims)'''
    
    for x in list_object:
        list1 = get_property_by_object(x[1][1][0], claims)
        for y in list1:
            if (y[1] == 'time'):
                try:
                    dt = convert_datetime_ISO8601(y[2][1:])
                    if (dt != False):
                        ###print('y[0], dt: ', y[0], dt)
                        list2.append([y[0], dt])
                except:
                    pass
        #print('list1###', list1)
        
        for z in list_property:
            try:
                temp_date = format_date_string(z[0])
                #print('temp_date: ', temp_date)
                temp = compare_datetime(temp_date, list2)
                if (temp != False):
                    temp1 = []
                    temp1 = get_dependency_nouns(sentence, z[0], z[4], [])
                    #print('z[0], z[4], sentence: ', z[0], z[4])
                    list3.append([z, [wikidataID] + [x[1][0][0]] + [x[1][1][0]] + temp, temp1])  
            except:
                continue

        #print('list1: ', list1)
        #print('list2: ', list2)

    #print('list3:++++++++++++++', list3)
    return list3

# find chunk by quantity items, ex: 20 => 20 goals
def find_chunk_from_quantity_root_head(list_property):

    items = []
    for p in list_property:
        if (p[3] == 'CARDINAL'):
            items.append(p)

    chunks = []
    for i in items:
        for p in list_property:
            if (i[1] == p[1] and i[2] < p[2] and p[3] == 'NOUN CHUNK'):
                chunks.append([i, p])

    return chunks            

# compare a number and a list
def compare_number(term, property_list):
    for p in property_list:
        print('.........', float(p[1]), float(term[0][0].replace('\'', '')), term[1][4], p[3])
        try:
            if (float(p[1]) == float(term[0][0].replace('\'', '')) and term[1][4] in p[3]):
                return [p[0], p[1]]      
        except:
            continue
    return False

# march property as a quantity (55 goals, 2 caps) 
def match_property_by_quantity(sentence, wikidataID, list_object, list_property, claims):
    list1 = [] 
    list2 = [] 
    list3 = [] 
    list4 = [] 

    #print("list_property.....", list_property)

    covers = find_chunk_from_quantity_root_head(list_property)
    #print("covers.....", covers)

    property_unit_list = property_unit()
    
    for x in list_object:
        list1 = get_property_by_object(x[1][1][0], claims)
        for y in list1:
            if (y[1] == 'quantity'):
                #print('y[0], y[2][1:]: ', y[0], y[2][1:])
                list2.append([y[0], y[2][1:]])
                
        #print("list2..........", list2)
        for z in list2:
            for p in property_unit_list:
                if (z[0] == p[0] and [z[0], z[1], p[1], p[2]] not in list3):
                    list3.append([z[0], z[1], p[1], p[2]]) # name, value, definition, units

        #list3 = list(set(list3))     
        #print("list3..........", list3)
        
        for t in covers:
            temp = compare_number(t, list3)
            print("t......", t)
            if (temp != False):
                temp1 = []
                temp1 = get_dependency_nouns(sentence, t[0][0], t[0][4], [])
                print('t[0][0], t[0][4], sentence: ', t[0][0], t[0][4])
                list4.append([t[0], [wikidataID] + [x[1][0][0]] + [x[1][1][0]] + temp, temp1])
   
    #print('list4:++++++++++++++', list4)
    return list4

# match property as a string
def match_property_by_string(sentence, wikidataID, list_object, list_property, claims):
    list1 = [] 
    list2 = [] 
    list3 = [] 
    
    for x in list_object:
        list1 = get_property_by_object(x[1][1][0], claims)
        for y in list1:
            if (y[1] == 'wikibase-entityid'):
                #print('y.....s', y)
                #get label of property
                root = get_wikidata_root(y[2])
                label = get_label(root)
                list2.append([y[0], label])

        for z in list_property:
            temp = compare_string(z, list2)
            if (temp != False):
                temp1 = []
                temp1 = get_dependency_nouns(sentence, z[0], z[4], [])
                print('z[0], z[4], sentence: ', z[0], z[4])
                list3.append([z, [wikidataID] + [x[1][0][0]] + [x[1][1][0]] + temp, temp1])
                
    #print('list3:++++++++++++++', list3)
    return list3

# compare a string and a list
def compare_string(dt, claims):
    for x in claims:
        try:
            if (x[1].lower() == dt[0].lower()): # convert to lowercase
                return [x[0], x[1]]        
        except:
            continue
    return False

# match property (qualifier)
def match_property(sentence, wikidataID, list_object, list_property, claims):   
    match1 = match_date_time(sentence, wikidataID, list_object, list_property, claims)
    match2 = match_property_by_quantity(sentence, wikidataID, list_object, list_property, claims)
    match3 = match_property_by_string(sentence, wikidataID, list_object, list_property, claims)
    return match1, match2, match3
    
# match object    
def match_object(sentence, wikidataID, entities, claims):
    return match_object_by_name(sentence, wikidataID, entities, claims)

# match object by name (label), only for r3 relations
def match_object_by_name(sentence, wikidataID, entities, claims):

    #print('entities:------', entities)
    #print('claims:------', claims)
    result_list = []
    for x in entities:
        for y in claims:
            if (x[0] in y[1]):
                #get_dependency_nouns(sentence, word, word_root)
                z = []
                #print('z, x[0], x[4]: ', z, x[0], x[4])
                z = get_dependency_nouns(sentence, x[0], x[4], [])
                #print('z, x[0], x[4]: ', z, x[0], x[4])
                if ([x, y, z] not in result_list):
                    result_list.append([x, y, z])

    #print('result list ne: ', result_list)
            # PUNKT system to detect abbreviation (future task)            
    return result_list

# get list of object name
def get_object_name_list(claims):
    result_list = []
    #print('List Objects: +++++++++', claims)
    for x in claims:
        sublist1 = []
        sublist2 = []
        id1 = x[1][1] # predicate wikidataID
        id2 = x[1][2] # object wikidataID

        #print('id1, id2: ', id1, id2)
        
        root1 = get_wikidata_root(id1)
        sublist1.append(id1)
        sublist1.append(get_label(root1)) # get label
        for x in get_alias(root1): # get aliases
            sublist1.append(x)  
        
        root2 = get_wikidata_root(id2)
        sublist2.append(id2)
        sublist2.append(get_label(root2)) # get label
        for y in get_alias(root2): # get aliases
            sublist2.append(y)
            
        result_list.append([sublist1, sublist2]) 
    return result_list
 
# match subject 
def match_subject(wikidataID, subject_name, subject_name_list):
    return match_human_name(wikidataID, subject_name, subject_name_list)
    
# get list of human name (subjects)
# P735-given_name, P734-family_name, Q49614-nick_name, P21-sex_or_gender: (he, she)
# alias? (future), occupation => the teacher, this teacher, etc (future)
def get_list_human_name(subject_name, claim_list):
    matching_list = []
    poss = ''
    matching_list.append(subject_name)
    for x in claim_list:
        for y in x:

            #print('y ne:', y)
            
            if (y[1]== 'P735' or y[1]== 'P734' or y[1]== 'Q49614'):
                root = get_wikidata_root(y[2])
                matching_list.append(get_label(root))
            if (y[1]== 'P21'):
                root = get_wikidata_root(y[2])
                sex = get_label(root)
                if (sex == 'male'):
                    matching_list.append('he')
                    poss = 'his'
                else:
                    matching_list.append('she')
                    poss = 'her'
    return matching_list, poss

# match human name
def match_human_name(wikidataID, subject_name, subject_name_list):
    if (subject_name[0].lower() in [x.lower() for x in subject_name_list]):
        return [subject_name, [wikidataID] + subject_name_list] # add subject (first item) to list
    return False

# get pages by category name (future)
def get_page_by_category(category_name):
    link = 'https://en.wikipedia.org/w/api.php?action=query&list=categorymembers&cmlimit=50&format=xml&cmtitle=' + category_name
    root = ''
    root = get_xml_data_by_url(link)
    if (root == ''):
        return ''

    page_list = []
    for node in root.find('./query/categorymembers'):
        if (node.tag == 'cm'):
            page_list.append((node.attrib['title']))
        
    return page_list
 
# match all claims (statements) of object 
def match_object_claims(wikidataID, object_wikidataID, object_name, entities):

    link = 'https://www.wikidata.org/w/api.php?action=wbgetentities&format=xml&ids=' + object_wikidataID
    root = get_xml_data_by_url(link)

    claim_list = get_claims(root, object_wikidataID) # claim_list
    #print('claim_list: ', claim_list)
    
    claim_list_r1 = filter_claim_by_type(claim_list, 'r1') # quantity
    #print('claim_list_r1: ', claim_list_r1)
    
    map_r1 = []
    if (len(claim_list_r1) > 0):
        for c in claim_list_r1:
            for e in entities:
                if (c[1][3].lower() == e[0].lower()): # string only (not datetime or quantity)
                    map_r1.append([wikidataID, object_wikidataID, object_name, c, e])
                    # match_property_by_quantity(sentence, wikidataID, list_object, list_property, claims)
    #print('map_r1: ', map_r1)
    
    claim_list_r2 = filter_claim_by_type(claim_list, 'r2') #wikibase-entityid
    #print('claim_list_r2: ', claim_list_r2)

    map_r2 = [] 
    if (len(claim_list_r2) > 0):
        for c in claim_list_r2:
  
            link1 = 'https://www.wikidata.org/w/api.php?action=wbgetentities&format=xml&ids=' + c[1][2]
            root1 = get_xml_data_by_url(link1)
                    
            label_list = get_alias(root1)
            label = get_label(root1)
            
            if (label != ''):
                label_list.append(label)
                temp_list1 = []

                # P31 - instance of
                if (c[1][1] == 'P31'): 
                    doc1 = nlp(label)
                    for chunk in doc1.noun_chunks:
                        temp_list1.append(chunk.root.text)

                #print('temp_list1: ', temp_list1)        
                for t in temp_list1:
                    label_list.append(t)

                # P27 - country of citizenship
                if (c[1][1] == 'P27'):
                    for item in country_list:
                        if (item[1].lower() == label.lower()): # search country
                            label_list.append(str(item[0])) # add nationality

            label_list = list(set(label_list)) #remove repetition                
            #print('label_list: ', label_list, object_wikidataID)
            
            for e in entities:
                #print('e ne: ', e)
                if (e[0].lower() in [x.lower() for x in label_list]):
                    #print('e, label_list: ', e, label_list)
                    map_r2.append([wikidataID, object_wikidataID, object_name, c, e])

    #print('map_r2: ', map_r2)
    return map_r1, map_r2

# get Wikidata hypernyms by level
def get_hypernyms(values, results, level):

    #print('values: ', values)
    #print('results: ', results)
    #print('~~~~~~~~~~~~~~~~~~~~~~')
    
    if (level == 0 or len(values) == 0):
        results += values
        results = [list(x) for x in set(tuple(x) for x in results)] # keep unique values
        return results

    terms = []
    for v in values:
        try:
            root = get_wikidata_root(v[0])
            claims = get_claims2(root, v[0])

            terms += get_values_by_property(claims, 'P31')
            terms += get_values_by_property(claims, 'P279')
        except:
            pass
        
    results += values
    results += terms
    results = [list(x) for x in set(tuple(x) for x in results)] # keep unique values

    terms = set(tuple(x) for x in terms)
    values = set(tuple(x) for x in values)
    terms = terms - values
    terms  = [list(x) for x in terms]

    #print('terms: ', terms)
    return get_hypernyms(terms, results, level - 1)


# check item in a list, return an entity
def get_item_entities(item, entities):
    for e in entities:
        try:
            if (item[0] == e[0] and item[1] == e[1] and item[2] == e[2]):
                return e
        except:
            pass
    return []
    
