from collections import Counter
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_lg')
from nltk import Tree
from collections import Counter
from nltk import ngrams

def tok_format(tok):
    return "_".join([tok.orth_])

def to_nltk_tree2(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(tok_format(node), [to_nltk_tree2(child) for child in node.children])
    else:
        return tok_format(node)

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

def ngram(sentence):
    list_result = []
    n = len(sentence)
    for i in range(0, n):
        list_result += ngrams(sentence.split(), i)
    return sorted(list(set(list_result)))

def get_dependency_nouns(sentence, word, word_root, exclude_list):

    doc = nlp(sentence)

    #named entities
    ent_list = []
    for ent in doc.ents:
        ent_list.append([ent.text, ent.label_])
    print("ent_list: ", ent_list)

    #noun chunks
    for chunk in doc.noun_chunks:
        print(chunk.text)

    # get custom stop words
    custom_stop_words = ["—", "-", "/", "|", ":"]
    for token in doc:
        #print(token.text, token.pos_, token.dep_)
        if (token.pos_ != "NOUN" and token.pos_ != "PROPN" and token.pos_ != "NUM" and token.text not in [x[0] for x in ent_list]):
            custom_stop_words.append(token.text)
    custom_stop_words = list(set(custom_stop_words))
    print("custom_stop_words: ", custom_stop_words)

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

            temp1 = []
            dependency_phrase(token.head, temp1)
            raw_list.append(temp1)
            
    print("raw_list", raw_list)

    # get dependency nouns/noun phrases
    ngram_list = []
    for x in raw_list:
        temp = ' '.join(str(e.strip()) for e in x)
        #print("temp, word:---", temp, "---",word)
        if (temp not in word and word not in temp):
            ngram_list.append(ngram(temp))

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

    print('exclude_list: ', exclude_list)        
    print('final_list2:', final_list)         

    try:
        [to_nltk_tree2(sent.root).pretty_print() for sent in doc.sents]
    except:
        print("Tree can not be displayed!")
    print("----------------------------------------------------")
    return final_list
    


#s = "In [i0], [s] married television producer [o0] in Las Vegas."
s = "In 2012 he married actress and singer Martine McCutcheon."

print(s)
print("----------------------------------------------------")
get_dependency_nouns(s, "actress", "actress", ["2012", "he", "Martine McCutcheon"])
