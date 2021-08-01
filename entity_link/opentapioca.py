import subprocess
import urllib.parse
import json
from wiki_core import *

try:
    from systems.utils.functions import url2id,url2id_wiki_normalized, ExistFile, position2numbertoken, position2numbertoken_doc, yago2wikiid
except:
    pass

try:
    from utils.functions import url2id,url2id_wiki_normalized, ExistFile, position2numbertoken, position2numbertoken_doc, yago2wikiid
except:
    pass

null = None

class OpenTapioca:
    url  = "https://opentapioca.org/api/annotate"
    number_of_request = 5
    key = ""
    lang = "EN"
    
    def __init__(self, l = "EN"):
        self.lang = l

    def request_curl(self, text):
        query_post = "query=" + urllib.parse.quote(text)
        for i in range(self.number_of_request):
            try:
                p = subprocess.Popen(['curl', '--data', query_post, self.url],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                stdout, stderr = p.communicate()
                if stdout:
                    self.raw_output = stdout
                    return stdout
            except Exception as err:
                print(err)
                continue
        return None

    def api(self, text):
        annotations = {}
        try:
            annotations = json.loads(self.request_curl(text))['annotations']
        except:
            pass
        
        doc = nlp(text)
        token_dict = {}
        for token in doc:
            token_dict[token.idx] = token.i
            token_dict[len(token.text) + token.idx] = token.i
    
        terms = []
        for a in annotations:
            try:
                start_char, end_char = a['start'], a['end']
                value = text[start_char:end_char]
                wikidata_id = a['best_qid']
                
                root = get_wikidata_root(wikidata_id)
                wiki_page = get_sitelink(root)
                
                start_token, end_token = token_dict[start_char], token_dict[end_char]     
                terms.append([value, start_token, end_token, start_char, end_char, {wikidata_id:wiki_page}])
            except:
                pass       
        print('terms: ', terms)
        return terms 


