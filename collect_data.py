#........................................................................................................
# Title: Wikidata claims (statements) to natural language (a part of Triple2Text/Ontology2Text task)
# Author: Ta, Hoang Thang
# Email: tahoangthang@gmail.com
# Lab: https://www.cic.ipn.mx
# Date: 12/2019
#........................................................................................................

from wiki_core import *
from read_write_file import *
import traceback

file_name = 'p26' # predicate property
file_name_output = 'output_' + file_name + '.csv'
page_list = read_from_csv_file('data/' + file_name + '.csv',  ',', 1)
page_list = sorted(list(set(page_list))) #filter repetitive pages and sort by alphabet
    
#create corpus file
with open(file_name_output, 'w', newline='', encoding='utf-8') as csvfile:
    fw = csv.writer(csvfile, delimiter='#', quoting=csv.QUOTE_MINIMAL)
    fw.writerow(['Type', 'Subject', 'Predicate', 'Object', 'Qualifier', 'Relation', 'Raw sentence', 'Subject', 'Subject matching', 'Root',
                             'Prepositional verb', 'Verbs', 'Predicate matching', 'Object', 'Object matching',
                             'Properties', 'Property matching', 'Labeled sentence 1', 'Labeled sentence 2', 'Order 1', 'Order 2'])
    csvfile.close()
    
# mapping statements to sentences by pages
for i, page in enumerate(page_list):
    print('page ', i, ': ', page)
    try:
        #root = get_xml_data_by_title('Henry Hoy')
        root = get_xml_data_by_title(page)
        wikidataID = get_wikidata_id(root) #wikidataID
        rootwikidata = get_wikidata_root(wikidataID)
        print(wikidataID, rootwikidata)
        
        text = html_content(root)
        
        #print(getcontentBySections(text))
        text = get_text_not_by_section(text) #text
        
        sentences = sentence_list(text)
        #print('sentences: ', sentences)

        claim_list = get_claims(rootwikidata, wikidataID) #claimList
        #print('claim_list: ', claim_list)

        #claim_list_r3 = filter_claim_by_type(claim_list, 'r3')
        #print('claim_list_r3: ', claim_list_r3)
        
        page_name = get_label(rootwikidata)
        #print('page_name: ', page_name)
        
        short_description = get_description(rootwikidata)

        #print('short_description: ', short_description)     
        analyze_sentence(sentences, page_name, wikidataID, claim_list, file_name_output)
        
        gc.collect() #remove cache
    except Exception as e:
        print('Error: ', e)
        traceback.print_exc()
        gc.collect() #remove cache
        #continue
    


