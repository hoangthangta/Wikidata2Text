import requests
import json
import urllib.parse

from wiki_core import *

class Babel():

    # babelfy_api
    def api(self, text, key='866555f7-dbe8-423c-a647-f4dfb637dcb9'):

        url = 'https://babelfy.io/v1/disambiguate?annRes=WIKI&th=.0&text=' + urllib.parse.quote(text, safe='')
        url += '&lang=EN&match=PARTIAL_MATCHING&source=WIKI&key=' + key
        print('url: ', url)
        
        response = requests.get(url)
        items = json.loads(response.text)

        terms = []
        for item in items:
            babelSynsetID = ''
            try:
                
                #babelSynsetID = item['babelSynsetID']
                #wiki_dict = self.babelnet_api(babelSynsetID)
                
                wiki_page = item['DBpediaURL'].split('/resource/')[1]
                root = get_xml_data_by_title(wiki_page)
                wikidata_id = get_wikidata_id(root)
            
                tokenFragment = item['tokenFragment']
                start_token, end_token = tokenFragment['start'], tokenFragment['end']
            
                charFragment = item['charFragment']
                start_char, end_char = charFragment['start'], charFragment['end']
                value = text[start_char:end_char + 1]

                terms.append([value, start_token, end_token, start_char, end_char, {wikidata_id:wiki_page}])
            except:
                pass
            
        print('terms: ', terms)
        return terms

    def babelnet_api(self, babel_synset_id, key='866555f7-dbe8-423c-a647-f4dfb637dcb9'):

        url = 'https://babelnet.io/v5/getSynset?id=' + babel_synset_id
        url += '&searchLang=EN&key=' + key

        response = requests.get(url)
        data = json.loads(response.text)

        senses = []
        try:
            senses = data['senses']
        except:
            pass

        wiki_dict = {}
        for sense in senses:
            properties = sense['properties']
            fullLemma =  remove_emojis(properties['fullLemma'])
            source = properties['source']
            senseKey = properties['senseKey']

            if (source == 'WIKIDATA'):
                #print(fullLemma, source, senseKey)

                if (senseKey not in wiki_dict):
                    wiki_dict[senseKey] = [fullLemma]
                else:
                    temp_list = wiki_dict[senseKey]
                    temp_list.append(fullLemma)
                    wiki_dict[senseKey] = temp_list
                    
        return wiki_dict
