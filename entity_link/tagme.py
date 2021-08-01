import requests
import json
import urllib.parse

from wiki_core import *

class Tagme():
    def api(self, text, key='b6556d72-cd68-475f-812f-5f8f1e7ef4af-843339462'):

        url = 'https://tagme.d4science.org/tagme/tag?lang=en&gcube-token=' + key + '&text=' + urllib.parse.quote(text, safe='')
        print('url: ', url)
        
        doc = nlp(text)
        token_dict = {}
        for token in doc:
            token_dict[token.idx] = token.i
            token_dict[len(token.text) + token.idx] = token.i

        response = requests.get(url)
        data = json.loads(response.text)

        annotations = []
        try:
            annotations = data['annotations']
        except:
            pass

        terms = []
        for annotation in annotations:
            try:
                wiki_page = annotation['title']
                root = get_xml_data_by_title(wiki_page)
                wikidata_id = get_wikidata_id(root)
            
                value = annotation['spot']
                start_char, end_char = annotation['start'], annotation['end']
                start_token, end_token = token_dict[start_char], token_dict[end_char]
            
                terms.append([value, start_token, end_token, start_char, end_char, {wikidata_id:wiki_page}])
            except:
                pass
            
        print('terms: ', terms)         
        return terms
