#........................................................................................................
# Title: Wikidata claims (statements) to natural language (a part of Triple2Text/Ontology2Text task)
# Author: Ta, Hoang Thang
# Email: tahoangthang@gmail.com
# Lab: https://www.cic.ipn.mx
# Date: 12/2019
#........................................................................................................

import csv

import spacy
from spacy.tokenizer import Tokenizer
nlp = spacy.load('en_core_web_md')

import sys
maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

# write file .txt
def write_to_text_file(file_name, data):
    with open(file_name, 'a', encoding='utf-8') as f:
        f.write(data + '\n')
    f.close()

# write file .txt
def write_to_new_text_file(file_name, data):
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(data + '\n')
    f.close()

# read file .txt
def read_from_text_file(file_name):
    page_list = []
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            for line in f:
              page_list.append(line.strip())
              #print(line)
            f.close()
    except:
        with open(file_name, 'a', encoding='utf-8') as f:
            f.close() 
    return page_list

# read from file csv
def read_from_csv_file(f, delimiter, column):
    page_list = []
    #print(f)
    with open(f, encoding='utf8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = delimiter) 
        for row in csv_reader:
            if (column == 'all'): #get all columns
                page_list.append(row)
            else:
                page_list.append(row[column])
    return page_list

# write to file csv
def write_to_csv_file(f, delimiter, list_data):
    with open(f, 'a', newline='', encoding='utf8') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter = delimiter, quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(list_data)
    csv_file.close()
    

# handle labeled items as a token in labeled sentences
def treat_labeled_items():
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
        infixes.remove(':')
    except Exception as e:
        pass
    
    try:
        infixes.remove('\\-')
    except Exception as e:
        pass

    try:
        infixes.remove('_')
    except Exception as e:
        pass

    try:
        infixes.remove(',')
    except Exception as e:
        pass
		
     
    infix_regex = spacy.util.compile_infix_regex(infixes)
    nlp.tokenizer = Tokenizer(nlp.vocab, infix_finditer=infix_regex.finditer)
	
# create corpus 1
def create_true_distribution_corpus1(page_list, start_row):

    corpus = []
    for p in page_list[start_row:]: # start row
        doc = nlp(p)   
        corpus.append([t.text for t in doc if t.text.strip() != ''])
		
    return corpus    

# create corpus 2 (for translation task)
def create_true_distribution_corpus2(sentence_list, start_row):
    treat_labeled_items()
    
    corpus = []
    for sentence in sentence_list[start_row:]: # start row
        doc = nlp('[start] ' + sentence + ' [end]')   
        corpus.append([t.text for t in doc if t.text.strip() != ''])
    
    return corpus  
