from wiki_core import *
import json

class DBpedia():

    def get_data_by_id(self, wikidata_id, format_type = 'json'):

        data, site_link = '', ''
        try:
            root = get_wikidata_root(wikidata_id)
            site_link = get_sitelink(root)
            
            if (site_link == ''): site_link = get_label(root) # get label instead of sitelink
            site_link = site_link.replace(' ', '_')
            
            link = 'http://live.dbpedia.org/data/' + site_link + '.' + format_type
            response = requests.get(link)
            #data = ET.fromstring(response.text)
            data = json.loads(response.text)
        except:
            pass
        
        return data, site_link

    def get_data_by_value(self, value, format_type = 'json'):

        data, value = '', ''
        try:
            value = value.replace(' ', '_').capitalize()
            link = 'http://live.dbpedia.org/data/' + value + '.' + format_type
            response = requests.get(link)

            #data = ET.fromstring(response.text)
            data = json.loads(response.text)
        except:
            pass
        return data, value

    def get_data_by_ontology_value(self, value, format_type = 'json'):

        data, value = '', ''
        try:
            value = value.replace(' ', '_').capitalize()
            link = 'http://live.dbpedia.org/data3/' + value + '.' + format_type

            response = requests.get(link)
            #data = ET.fromstring(response.text)
            data = json.loads(response.text)
        except:
            pass
        return data, value

    def get_subclass_ontology(self, data, site_link):

        types = []
        try:
            key = 'http://live.dbpedia.org/ontology/' + site_link 
            dict_types = data[key]['http://www.w3.org/2000/01/rdf-schema#subClassOf']
            for d in dict_types:
                value = d['value'].replace('http://live.dbpedia.org/ontology/', '').lower()

                if ('http://www.w3.org/2002/07/owl#' in value):
                        value = value.replace('http://www.w3.org/2002/07/owl#', '')

                if (value == 'thing'): continue # avoid 'root' ontology
                
                types.append(value)
        except:
            pass
        
        return types
        
    def get_type(self, data, site_link):

        types = []
        try:
            key = 'http://live.dbpedia.org/resource/' + site_link
            print('key: ', key)
            dict_types = data[key]['http://www.w3.org/1999/02/22-rdf-syntax-ns#type']
            print('dict_types: ', dict_types)
            
            for t in dict_types:
                value = t['value']
                if ('http://live.dbpedia.org/ontology/' in value):
                    value = value.replace('http://live.dbpedia.org/ontology/', '')
                    types.append(value.lower())
        except:
            pass
        
        return types

    def get_hypernyms(self, values, results, level):

        if (level == 0 or len(values) == 0):
            temp = []
            try:
                temp = list(set(values + results))
            except:
                pass
            return temp
            
        terms = []
        for v in values:
            data, site_link = self.get_data_by_ontology_value(v)
            terms += self.get_subclass_ontology(data, site_link)
        
        results += values
        results += terms
        results = list(set(results))

        terms = set(terms)
        values = set(values)

        terms = list(terms - values)

        #print('terms: ', terms)
        return self.get_hypernyms(terms, results, level - 1)

        
#db = DBpedia()
#data, site_link = db.get_data_by_ontology_value('Person')
#types = db.get_subclass_ontology(data, site_link)
#types = db.get_hypernyms(['Person','Agent'],[],1)
#print(types)
