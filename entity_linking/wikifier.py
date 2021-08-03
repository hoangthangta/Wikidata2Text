import requests
import json
import urllib.parse

from wiki_core import *

class Wikifier():
        
    def api(text, key='almmyawmnxyoytgnlkmnjasnrgnjcv'):
        url = 'http://www.wikifier.org/annotate-article?lang=en&text=' + urllib.parse.quote(text, safe='')
        url += '&wikiDataClasses=false&userKey=' + key
        print('url: ', url)

        terms = []
        response = requests.get(url)
        items = json.loads(response.text)['annotations']

        for item in items:
            title = item['title']

            wikidata_id = ''
            try:
                wikidata_id = item['wikiDataItemId']
            except Exception as e:
                #print('Loi: ', e)
                pass
            
            support = item['support']
            start_token, end_token = support[0]['wFrom'], support[0]['wTo']
            start_char, end_char = support[0]['chFrom'], support[0]['chTo']
            value = text[start_char:end_char+1]

            terms.append([value, start_token, end_token, start_char, end_char, {wikidata_id:title}])

        print(terms)
        return terms  
