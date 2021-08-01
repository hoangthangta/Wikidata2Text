#........................................................................................................
# Title: Wikidata claims (statements) to natural language (a part of Triple2Text/Ontology2Text task)
# Author: Ta, Hoang Thang
# Email: tahoangthang@gmail.com
# Lab: https://www.cic.ipn.mx
# Date: 12/2019
#........................................................................................................

from collections import Counter
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_lg')

# average of page length
def page_length_average(page_list):
    count_list = []
    for p in page_list:
        count_list.append([len(p.split()), p])
    return Counter([x[0] for x in count_list])

#get a distance list of tokens by sentences from different 2 items in tuples (s, o, i) or (i, s, o)
def list_distance_frequency(item1, item2, page_list):
    list_ = []
    for p in page_list:
        temp_list = p[14].split()
        index1 = -1
        index2 = -1
        dist = -1
        for i, val in enumerate(temp_list):
            if (val==item1):
                index1 = i
            if (val==item2):
                index2 = i
        dist = abs(index1 - index2)       
        if (dist > 0):
            list_.append([dist, p])
    return list_

#filter page list by an increase percent with threshold = 0.1 (or 10%) 
def filter_by_threshold(list_rev, total):
    t = 0
    s = list_rev[0]/total
    list_n = []
    print("No.", "Total\t" , "Percent\t\t", "Increase percent\t\t", "Previous increase percent")
    
    for x in list_rev:
        t += x
        print(x, total, "\t\t", t/total, "\t\t", ((t/total) - s)/s, "\t\t", s)
        list_n.append([x, t/total, ((t/total) - s)/s, s])
        s = t/total
        
    list_r = []
    for i, val in enumerate(list_n):
        if (val[2] > 0.1 or i == 0):
            list_r.append(val[0])
    
    return list_r

#calculate the mean of distances
def calculate_mean(list_r, counter):
    list_m = []
    for x in counter:
        key = x
        value = counter[key]
        if (value in list_r):
            list_m.append(key)
    print("list_m: ", list_m)        
    return mean(list_m)

#get the largest distance in group
def calculate_largest_distance(list_r, counter):
    list_m = []
    for x in counter:
        key = x
        value = counter[key]
        if (value in list_r):
            list_m.append(key)   
    return max(list_m)
        
#remove noises based on distances and an increase percent with threshold = 0.1 (or 10%)  (future)
def remove_noise(page_list):
    list1 = list_distance_frequency("[s]", "[o0]", page_list)
    
    total = len(list1)
    print("total: ", total)
    
    ct1 = Counter([x[0] for x in list1])
    print(ct1)
    
    list_rev = {v:k for k, v in ct1.items()}
    list_rev = sorted(list_rev.keys(), reverse=True)
    print("list_rev:", list_rev)

    list_r = filter_by_threshold(list_rev, total)

    largest_dist = calculate_largest_distance(list_r, ct1)
  
    list_l = []
    for x in list1:
        if (x[0] <= largest_dist):
            #list_l.append(x[1][14])
            list_l.append(x[1])
    return largest_dist, list_l

# remove redundant words based on dependency parsing
def remove_redundant_words(page_list):

    stop_words = ["[s]", "[o0]", "[o1]", "[o2]", "[o3]", "[i:"] #lỗi chỗ này
    results = []

    for p in page_list:
        list_words = []      
        delete_words = []
        doc = nlp(p[2]) #raw sentence
        options = {"compact":True}
        displacy.render(doc, style="dep", options = options)

        #check for redundant words
        for token in doc:
            list_words.append(token.text)

            if (token.dep_ == "amod" and token.i != 0): #adjective modifier
                if (token.pos_ == "PROPN"): #pass if pronoun
                    continue
                if (token.text[0].isupper() == True): #pass if pronoun, case of Japanese
                    continue
                delete_words.append([token.text, token.i])
                
            if (token.dep_ == "advmod" and token.i != 0): #adverb modifier
                delete_words.append([token.text, token.i])           
            if (token.dep_ == "nummod" and token.i != 0): #number modifier
                delete_words.append([token.text, token.i])          
            #print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)
            #print(token.dep_, "(", token.head, "-", token.head.i+1, ",", token.text, "-", token.i+1, ")", sep="")

        result_words = p[13].split() #p13 has ~ tokens with p2
       
        for x in delete_words:
            for i2, val2 in enumerate(list_words):     
                try: # pass error with ; in a sentence
                    if (x[1] == i2):
                        result_words[i2] = ""
                except:
                    continue

        result_words2 = [] 
        for i, val in enumerate(result_words):
            temp = result_words[i]
            if (len(result_words2)==0):
                result_words2.append(temp)
            elif (temp != result_words2[len(result_words2)-1] and temp in stop_words):
                result_words2.append(temp)
            elif (temp not in stop_words):
                result_words2.append(temp)
                
        s = ' '.join(str(e.strip()) for e in result_words2 if e!='')

        results.append(s)
    return results       
