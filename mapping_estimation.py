#........................................................................................................
# Title: Wikidata claims (statements) to natural language (a part of Triple2Text/Ontology2Text task)
# Author: Ta, Hoang Thang
# Email: tahoangthang@gmail.com
# Lab: https://www.cic.ipn.mx
# Date: 12/2019
#........................................................................................................

from base import *
from wiki_core import *
from dbpedia_core import *

from entity_linking.opentapioca import OpenTapioca
from entity_linking.aida import Aida
from entity_linking.wat import Wat
from entity_linking.wikifier import Wikifier
from entity_linking.tagme import Tagme
from entity_linking.babel import Babel

import requests
import json
import ast
import urllib.parse

# https://spacy.io/api/annotation
named_entities = {'ORG':['organization', 'organisation', 'company', 'agency', 'institution', 'agent'],
                  'GPE':['country', 'city', 'state', 'geopolitical entity', 'political party'],
                  'PERSON':['human', 'person', 'he', 'she'],
                  'MONEY':['money', 'currency', 'monetary value'],
                  'NORP':['nationality', 'religion', 'political group'],
                  'FAC':['building', 'airport', 'highway', 'bridge'],
                  'WORK_OF_ART':['book', 'song', 'work'],
                  'DATE':['date', 'time'],
                  'TIME':['time', 'date'],
                  'PRODUCT':['product', 'object', 'vehicle', 'food'],
                  'LOC':['loc', 'mountain range', 'body of water'],
                  'EVENT':['event', 'hurricane', 'battle', 'war', 'sport event'],
                  'LAW':['law', 'document'],
                  'LANGUAGE':['language'],
                  'PERCENT':['percent', 'percentage'],
                  'QUANTITY':['quantity', 'weight', 'distance','measurement'],
                  'ORDINAL':['ordinal'],
                  'CARDINAL':['number']
                  }

# evaluate subject matching
def evaluate_subject_matching_type(df, method='wikidata', depth=0):

    total_score = 0
    total_row = len(df.index)

    if (total_row == 0):
        return 0, 0, 0

    no_hypernym_rows = 0
    no_subject_rows = 0
    current_row = 0
    for index, row in df.iterrows():

        print('....................................................')
        print('....................................................')

        current_row += 1
        sentence = row['raw_sentence']
        print('sentence: ', sentence)
        
        entity_list = get_entities2(sentence)
        print('entity_list: ', entity_list)
        
        subject_values = []
        subject_matchings = []
        try:
            subject_values = [x.strip() for x in row['subject_value'].split('|')] # split by '|'
            subject_values = [ast.literal_eval(x) for x in subject_values]
            
            subject_matchings = [x.strip() for x in row['subject_matching'].split('|')] # split by '|'
            subject_matchings = [ast.literal_eval(x)[0].strip().strip("'") for x in subject_matchings]
        except:
            pass

        #print('subject_values: ', subject_values, row['subject_value'])
        #print('subject_matchings: ', subject_matchings, row['subject_matching'])

        length = len(subject_values)
        if (length == 0):
            no_subject_rows += 1
            score = 0
            print('total_score, current_row, no_hypernym_rows,  no_subject_rows, score, length: ', total_score,
                              current_row, no_hypernym_rows, no_subject_rows, score, length)
            write_to_text_file('data/subject_matching_type_' + method + '_' + str(depth) + '.txt', '--no subject value-- ' + sentence + ',' + str(total_score)
                               + ',' + str(current_row) + ',' + str(no_hypernym_rows) + ',' + str(no_subject_rows) + ',' + str(score) + ',' + str(length))
            continue

        score = 0
        local_hypernym_rows = 0
        for value, wikidata_id in zip(subject_values, subject_matchings):

            ne = ''
            if (value[0].lower() in ['he', 'she']): ne = 'PERSON'
            else:
                temp = get_item_entities(ne, entity_list)
                if (len(temp) != 0): ne = temp[3]  
            
            hypernyms = []
            hypernym_values = []
            if (method == 'wikidata'):
                print('wikidata_id: ', wikidata_id)
                root = get_wikidata_root(wikidata_id)
                claims = get_claims2(root, wikidata_id)
                values = []
                values += get_values_by_property(claims, 'P31') # instance of
                values += get_values_by_property(claims, 'P279') # subclass of
                hypernyms = get_hypernyms(values, [], depth) 
                hypernym_values = [x[1] for x in hypernyms]
                print('hypernym_values: ', hypernym_values)
            elif (method == 'dbpedia'):
                db = DBpedia()
                data, site_link = db.get_data_by_id(wikidata_id)
                types = db.get_type(data, site_link)
                hypernym_values = db.get_hypernyms(types, [], depth)             
                print('hypernym_values: ', hypernym_values)

            # no hypernyms
            if (len(hypernym_values) == 0):
                local_hypernym_rows += 1

            nes = []
            if (ne in named_entities):
                nes = named_entities[ne]
            print('nes: ', nes)
            
            intersection = set(hypernym_values).intersection(set(nes))
            if (len(intersection) != 0):
                print('**********************')
                print('named entity type, wikidata_id, hypernym_values, nes, intersection: ', ne, wikidata_id, hypernym_values,
                      nes, intersection)
                print('total_score, current_row, no_hypernym_rows,  no_subject_rows, score, length: ', total_score,
                              current_row, no_hypernym_rows, no_subject_rows, score, length)
                score += 1
                print('**********************')

        if (local_hypernym_rows != 0):
            no_hypernym_rows += local_hypernym_rows/length
        total_score += score/length
        print('----------------------------')
        print('total_score, current_row, no_hypernym_rows,  no_subject_rows, score, length: ', total_score,
                              current_row, no_hypernym_rows, no_subject_rows, score, length)
        write_to_text_file('data/subject_matching_type_' + method + '_' + str(depth) + '.txt', sentence + ',' + str(total_score)
                               + ',' + str(current_row) + ',' + str(no_hypernym_rows) + ',' + str(no_subject_rows) + ',' + str(score) + ',' + str(length))


# evaluate qualifier matching
def evaluate_qualifier_matching_type(df, method='wikidata', depth=0):

    total_score = 0
    total_row = len(df.index)

    if (total_row == 0):
        return 0, 0, 0
    #print('total_score, total_row: ', total_score, total_row)
    
    no_datatype_rows = 0 # the number of rows can not get datatypes from Wikidata
    no_qualifier_rows = 0
    current_row = 0
    for index, row in df.iterrows():

        print('....................................................')
        print('....................................................')

        current_row += 1
        sentence = row['raw_sentence']
        print('sentence: ', sentence)
        
        qualifier_values = []
        qualifier_matchings = []
        try:
            qualifier_values = [x.strip() for x in row['qualifier_value'].split('|')] # split by '|'
            qualifier_values = [ast.literal_eval(x)[3].strip().strip("'") for x in qualifier_values]
            
            qualifier_matchings = [x.strip() for x in row['qualifier_matching'].split('|')] # split by '|'
            qualifier_matchings = [ast.literal_eval(x)[3].strip().strip("'") for x in qualifier_matchings]
        except:
            pass

        #print('qualifier_values: ', qualifier_values, row['qualifier_value'])
        #print('qualifier_matchings: ', qualifier_matchings, row['qualifier_matching'])

        length = len(qualifier_values)
        if (length == 0):
            score = 0
            no_qualifier_rows += 1
            print('total_score, current_row, ,  no_datatype_rows, score, length: ', total_score,
                              current_row, no_datatype_rows, no_qualifier_rows, score, length)
            write_to_text_file('data/qualifier_matching_type_' + method + '_' + str(depth) + '.txt', '--no qualifier values-- ' + sentence + ',' + str(total_score)
                               + ',' + str(current_row) + ',' + str(no_datatype_rows) + ',' + str(no_qualifier_rows) + ',' + str(score) + ',' + str(length))
            continue
        
        score = 0
        local_datatype_rows = 0
        
        for ne, wikidata_id in zip(qualifier_values, qualifier_matchings):

            datatype = ''
            if (method == 'wikidata'):
                root = get_wikidata_root(wikidata_id)
                datatype = get_property_datatype(root)
                print('ne, wikidata_id, datatype: ', ne, wikidata_id, datatype)
            elif (method == 'dbpedia'):
                db = DBpedia()
                print('wikidata_id:' , wikidata_id)
                data, site_link = db.get_data_by_id(wikidata_id)
                print('data, site_link: ', data, site_link)
                types = db.get_type(data, site_link)
                print('types: ', types)
                hypernym_values = db.get_hypernyms(types, [], depth)             
                print('hypernym_values: ', hypernym_values)
                datatype = hypernym_values
                
            # no wikidata results
            if (datatype == ''):
                local_datatype_rows += 1
                
            nes = []
            if (ne in named_entities):
                nes = named_entities[ne]

            #intersection = set(hypernym_values).intersection(set(nes))
            if (datatype in nes):
                print('**********************')
                print('named entity, wikidata_id, datatype, nes: ', ne, wikidata_id, datatype, nes)
                print('total_score, current_row, no_datatype_rows,  no_qualifier_rows, score, length: ', total_score,
                              current_row, no_datatype_rows, no_qualifier_rows, score, length)
                score += 1
                print('**********************')

        if (local_datatype_rows != 0):
            no_datatype_rows += local_datatype_rows/length
        total_score += score/length
        
        print('----------------------------')
        print('total_score, current_row, no_datatype_rows,  no_qualifier_rows, score, length: ', total_score,
                              current_row, no_datatype_rows, no_qualifier_rows, score, length)
        write_to_text_file('data/qualifier_matching_type_' + method + '_' + str(depth) + '.txt', sentence + ',' + str(total_score)
                               + ',' + str(current_row) + ',' + str(no_datatype_rows) + ',' + str(no_qualifier_rows) + ',' + str(score) + ',' + str(length))
                               

            
# evaluate object matching
def evaluate_object_matching_type(df, method='wikidata', depth=0):

    total_score = 0
    total_row = len(df.index)

    if (total_row == 0):
        return 0, 0, 0
    
    #print('total_score, total_row: ', total_score, total_row)
    no_hypernym_rows = 0
    no_object_rows = 0
    current_row = 0
    for index, row in df.iterrows():

        print('....................................................')
        print('....................................................')

        current_row += 1
        sentence = row['raw_sentence']
        print('sentence: ', sentence)
        
        object_values = []
        object_matchings = []
        try:
            object_values = [x.strip() for x in row['object_value'].split('|')] # split by '|'
            object_values = [ast.literal_eval(x)[3].strip().strip("'") for x in object_values]
            
            object_matchings = [x.strip() for x in row['object_matching'].split('|')] # split by '|'
            object_matchings = [ast.literal_eval(x)[0].strip().strip("'") for x in object_matchings]
        except:
            pass
        print('object_values: ', object_values, row['object_value'])
        print('object_matchings: ', object_matchings, row['object_matching'])

        length = len(object_values)
        if (length == 0):
            score = 0
            no_object_rows += 1
            print('total_score, current_row, no_hypernym_rows,  no_object_rows, score, length: ', total_score,
                              current_row, no_hypernym_rows, no_object_rows, score, length)
            write_to_text_file('data/object_matching_type_' + method + '_' + str(depth) + '.txt', '--no object values-- ' + sentence + ',' + str(total_score)
                               + ',' + str(current_row) + ',' + str(no_hypernym_rows) + ',' + str(no_object_rows) + ',' + str(score) + ',' + str(length))
            continue
        
        score = 0 # local_score
        local_hypernym_rows = 0
        # named entity, wikidata_id
        for ne, wikidata_id in zip(object_values, object_matchings):

            #if (ne == 'NOUN CHUNK'): continue
            hypernyms = []
            hypernym_values = []
            if (method == 'wikidata'):
                print('wikidata_id: ', wikidata_id)
                root = get_wikidata_root(wikidata_id)
                claims = get_claims2(root, wikidata_id)
                values = []
                values += get_values_by_property(claims, 'P31') # instance of
                values += get_values_by_property(claims, 'P279') # subclass of
                hypernyms = get_hypernyms(values, [], depth)
                hypernym_values = [x[1] for x in hypernyms]
                print('hypernym_values: ', hypernym_values)
            elif (method == 'dbpedia'):
                db = DBpedia()
                data, site_link = db.get_data_by_id(wikidata_id)
                types = db.get_type(data, site_link)
                hypernym_values = db.get_hypernyms(types, [], depth)             
                print('hypernym_values: ', hypernym_values)

            # no hypernyms
            if (len(hypernym_values) == 0):
                local_hypernym_rows += 1

            nes = []
            if (ne in named_entities):
                nes = named_entities[ne]

            intersection = set(hypernym_values).intersection(set(nes))
            if (len(intersection) != 0):
                print('**********************')
                print('named entity type, wikidata_id, nes, intersection: ', ne, wikidata_id, hypernym_values, nes, intersection)
                print('total_score, current_row, no_hypernym_rows,  no_object_rows, score, length: ', total_score,
                              current_row, no_hypernym_rows, no_object_rows, score, length)
                score += 1
                print('**********************')

        if (local_hypernym_rows != 0):
            no_hypernym_rows += local_hypernym_rows/length
        total_score += score/length
        print('----------------------------')
        print('total_score, current_row, no_hypernym_rows,  no_object_rows, score, length: ', total_score,
                              current_row, no_hypernym_rows, no_object_rows, score, length)
        write_to_text_file('data/object_matching_type_' + method + '_' + str(depth) + '.txt', sentence + ',' + str(total_score)
                               + ',' + str(current_row) + ',' + str(no_hypernym_rows) + ',' + str(no_object_rows) + ',' + str(score) + ',' + str(length)) # replace by "#" for better

    #return total_score, total_row,  total_score/total_row        

# evaluate object matching by methods: wikifier, babelfy, tagme, wat, and aida.
def evaluate_object_matching(method, dataset):
    total_score = 0
    total_rows = len(dataset.index)

    current_rows = 0
    no_term_rows = 0 # rows have no terms
    match_rows = 0 # rows match terms 

    for index, row in dataset.iterrows():
        
        print('....................................................')
        print('....................................................')

        current_rows += 1
        sentence = row['raw_sentence']
        print('sentence: ', sentence)

        # terms
        terms = []

        if (method == 'wikifier'):
            et = Wikifier()
            terms = et.api(sentence)
        elif(method == 'babelfy'):
            et = Babel()
            terms = et.api(sentence)
        elif(method == 'tagme'):
            et = Tagme()
            terms = et.api(sentence)
        elif(method == 'wat'):
            et = Wat()
            terms = et.api(sentence)
        elif (method == 'aida'):
            et = Aida()
            terms = et.api(sentence)
        elif (method == 'opentapioca'):
            et = OpenTapioca()
            terms = et.api(sentence)
            

        print('terms: ', terms, method)

        if (len(terms) == 0):
            no_term_rows += 1
            write_to_text_file('data/object_matching_' + method + '.txt', '--no terms-- ' + sentence + ',' + str(total_score) + ',' + str(current_rows) + ',' +
                           str(match_rows) + ',' + str(no_term_rows) + ',' + str(current_rows - match_rows - no_term_rows))
            continue

        object_values = []
        object_matchings = []
        try:
            object_values = [ast.literal_eval('[' + item.strip('"') + ']') for item in row['object_value'].split('|')]
            object_matchings = [ast.literal_eval('[' + item.strip('"') + ']') for item in row['object_matching'].split('|')]
        except Exception as e:
            pass

        local_length = len(object_values)
        for text_item, wikidata_item in zip(object_values, object_matchings):    
            for t in terms:
                print('---------------------------------------')
                print('-- current term: ', t)
                print('-- object_values: ', object_values)
                print('-- object_matchings: ', object_matchings)

                if (t[0] in [text_item[0], text_item[4]] and (t[1] in range(text_item[1]-2, text_item[2]+2))
                    and (t[2] in range(text_item[1]-2, text_item[2]+2)) and (wikidata_item[0] in t[5])):
                    
                    print('-- local score: ', 1/(len(t[5])*local_length), len(t[5]))
                    print('total_score, current_rows, match_rows, no_term_rows, no_match_rows: ', total_score, current_rows,
                          match_rows, no_term_rows, current_rows - match_rows - no_term_rows)
                    
                    total_score += 1/(len(t[5])*local_length)

                    match_rows += 1
                    break

        print('total_score, current_rows, match_rows, no_term_rows, no_match_rows: ', total_score, current_rows,
                          match_rows, no_term_rows, current_rows - match_rows - no_term_rows)
        print('local_length:', local_length)
        
        write_to_text_file('data/object_matching_' + method + '.txt', sentence + ',' + str(total_score) + ',' + str(current_rows) + ',' +
                           str(match_rows) + ',' + str(no_term_rows) + ',' + str(current_rows - match_rows - no_term_rows))
    

# evaluate subject matching: wikifier, babelfy, tagme, wat, and aida.
def evaluate_subject_matching(method, dataset, mode='untrained'):

    total_score = 0
    total_rows = len(dataset.index)

    current_rows = 0
    no_term_rows = 0 # rows have no terms
    match_rows = 0 # rows match terms 

    for index, row in dataset.iterrows():

        print('....................................................')
        print('....................................................')
        
        current_rows += 1
        sentence = row['raw_sentence']
        print('sentence: ', sentence)

        # trained sentence
        trained_sentence, root, subject = '', '', ''
        if (mode == 'trained'):
            trained_sentence = row['trained_sentence']
            print('trained_sentence: ', trained_sentence)
            
            root = get_root(trained_sentence)
            if (root[0] != ''):
                try:
                    subject = get_subject(trained_sentence, root)
                except:
                    pass
            else: root = ''  
        
        # terms
        terms = []
        text = ''
        if (mode == 'trained'):
            text = trained_sentence
        else:
            text = sentence
            
        if (method == 'wikifier'):
            et = Wikifier()
            terms = et.api(sentence)
        elif(method == 'babelfy'):
            et = Babel()
            terms = et.api(sentence)
        elif(method == 'tagme'):
            et = Tagme()
            terms = et.api(sentence)
        elif(method == 'wat'):
            et = Wat()
            terms = et.api(sentence)
        elif (method == 'aida'):
            et = Aida()
            terms = et.api(sentence)
        elif (method == 'opentapioca'):
            et = OpenTapioca()
            terms = et.api(sentence)

        print('terms: ', terms, method)

        if (len(terms) == 0):
            no_term_rows += 1
            write_to_text_file('data/subject_matching_' + mode + '_' + method + '.txt', '--no terms-- ' + sentence + ',' + str(total_score) + ',' + str(current_rows) + ',' +
                           str(match_rows) + ',' + str(no_term_rows) + ',' + str(current_rows - match_rows - no_term_rows))
            continue

        subject_value = []
        subject_matching = []
        try:
            subject_value = ast.literal_eval("[" + row['subject_value'] + "]")
            subject_matching = ast.literal_eval("[" + row['subject_matching'] + "]")    
        except Exception as e:
            pass

        #print('subject_value: ', subject_value)
        #print('subject_matching: ', subject_matching)
 
        for t in terms:
            print('---------------------------------------')
            print('-- current term: ', t)
            print('-- subject_value: ', subject_value)
            print('-- subject_matching: ', subject_matching)
            print('-- root: ', root)
            print('-- subject: ', subject)

            if (mode == 'trained'):

                if (subject == ''): # break if can not extract subject
                    break
                
                if (t[0] == subject[0] and (t[1] in range(subject[1]-2, subject[2]+2))
                    and (t[2] in range(subject[1]-2, subject[2]+2)) and (subject_matching[0] in t[5])
                    and (subject[1] in range(subject_value[1]-2, subject_value[1]+2))):
                    
                    print('-- local score: ', 1/(len(t[5])), len(t[5]))
                    print('total_score, current_rows, match_rows, no_term_rows, no_match_rows: ', total_score, current_rows,
                            match_rows, no_term_rows, current_rows - match_rows - no_term_rows)
                    
                    total_score += 1/(len(t[5]))

                    match_rows += 1
                    break

            else:
                if (t[0] == subject_value[0] and (t[1] in range(subject_value[1]-2, subject_value[2]+2))
                    and (t[2] in range(subject_value[1]-2, subject_value[2]+2)) and (subject_matching[0] in t[5])):
                    
                    print('-- local score: ', 1/(len(t[5])), len(t[5]))
                    print('total_score, current_rows, match_rows, no_term_rows, no_match_rows: ', total_score, current_rows,
                            match_rows, no_term_rows, current_rows - match_rows - no_term_rows)
                    
                    total_score += 1/(len(t[5]))

                    match_rows += 1
                    break

        print('total_score, current_rows, match_rows, no_term_rows, no_match_rows: ', total_score, current_rows,
                          match_rows, no_term_rows, current_rows - match_rows - no_term_rows)
        
        write_to_text_file('data/subject_matching_' + mode + '_' + method + '.txt', sentence + ',' + str(total_score) + ',' + str(current_rows) + ',' +
                           str(match_rows) + ',' + str(no_term_rows) + ',' + str(current_rows - match_rows - no_term_rows))
                              
# print all sentences
def print_sentences(df):
    for index, row in df.iterrows():
        sentence = row['raw_sentence']
        print(sentence)

