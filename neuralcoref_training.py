from base import *
import spacy
import neuralcoref

from wiki_core import *

nlp = spacy.load('en_core_web_md')
neuralcoref.add_to_pipe(nlp)
nlp.add_pipe(nlp.create_pipe('sentencizer'), before='parser')

# train by neuralcoref
def train_content_by_neuralcoref(sents_list, sentence_pos):
    content = ' '.join(r.strip() for r in sents_list)
    doc = nlp(content)
    trained_clusters = doc._.coref_clusters     # [My sister: [My sister, She], a dog: [a dog, him]]
    trained_content = doc._.coref_resolved      # 'My sister has a dog. My sister loves a dog.'
    
    doc = nlp(trained_content)
    trained_sentence_list = []
    for sent in doc.sents:
        trained_sentence_list.append(sent.text.strip())

    #print('sentence_list: ', len(sents_list))
    #print('trained_sentence_list: ', len(trained_sentence_list))
    #print('sentence_pos: ', sentence_pos)
    
    trained_sentence = ''
    if (len(trained_sentence_list) == len(sents_list) and sentence_pos in range(0, len(trained_sentence_list))):
        trained_sentence = trained_sentence_list[sentence_pos]

    return trained_sentence

# get trained sentence by neuralcoref (for evaluating subject matching)
def get_trained_sentence(df):
    for index, row in df.iterrows():

        print('.................................')
        print('.................................')
        sentence = row['raw_sentence']
        subject_id = row['subject'] # wikidata_id of subject

        subject_value = row['subject_value']
        print('subject_value :', subject_value)

        wikidata_root = get_wikidata_root(subject_id)
        page_name = get_sitelink(wikidata_root)
        
        article_root = get_xml_data_by_title(page_name)
        text = html_content(article_root)

        text = get_text_not_by_section(text)

        # get sentences
        doc = nlp(text)
        sents_list = []
        for sent in doc.sents:
            sents_list.append(sent.text.strip())

        pos = -1   
        for k, value in enumerate(sents_list):
            if (sentence == value):
                pos = k

        #print('pos: ', pos)
        
        trained_sentence = '' 
        if (pos != -1):
            trained_sentence = train_content_by_neuralcoref(sents_list, pos)
        
        print('subject_id, sitelink:', subject_id, page_name)
        print('sentence before: ', sentence)
        print('sentence after: ', trained_sentence)

        if (trained_sentence == ''):
            trained_sentence = sentence

        row['trained_sentence'] = trained_sentence

        print('row', row.to_list())
        write_to_csv_file('data/output_common2_trained.csv', '#', row.to_list())

