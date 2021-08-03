# Wikidata2Text

Wikidata2Text (Wikidata statements to Text - WS2T) is a sub-task of Data2Text. This task is to translate Wikidata claims or statements, organized as a set of triples or quadruples to Wikipedia sentences.  

Our paper are currently under review at http://www.semantic-web-journal.net. Please check the guideline to follow when we publish our paper. The code is mainly about the data mapping process.

# Data
We have two folders for the data:
* `Our_data`: is the folder containing all of our data for the paper.
* `Data`: this folder is for the experiment, we already set some data files for testing purposes here.

# How to collect data?
At first, we have to collect the pairs of `(item, page)` via Wikidata query server at https://query.wikidata.org/. We use 2 queries:

**Query 1** - Retrieve all qualifiers of property P26 and count their occurrence frequency.
```
SELECT ?qual ?qualLabel ?count WHERE {
 {
 SELECT ?qual (COUNT(DISTINCT ?item) AS ?count) WHERE {
 hint:Query hint:optimizer "None" .
 ?item p:P26 ?statement .
 ?statement ?pq_qual ?pq_obj .
 ?qual wikibase:qualifier ?pq_qual .
 }
 GROUP BY ?qual
 } .
 OPTIONAL {
 ?qual rdfs:label ?qualLabel filter (lang(?qualLabel) = "en") .
 }
}
ORDER BY DESC(?count) ASC(?qualLabel)
```

**Query 2** - Capture a page list by qualifiers (P580, P582) of property P26. We can download the results of this file as *.csv format.
```
SELECT ?item ?title ?object ?property ?value ?sitelink WHERE {
 ?item p:P26 ?object.
 ?object ps:P26 ?property;
 pq:P580|pq:P582 ?value.
 ?sitelink schema:about ?item;
 schema:isPartOf <https://en.wikipedia.org/>;
 schema:name ?title.
 SERVICE wikibase:label { bd:serviceParam wikibase:language "en,en". }
}
LIMIT 50000
```

In this project, we store several files for Wikidata properties as following:
- `data/p108.csv`
- `data/p166.csv`
- `data/p26.csv`
- `data/p39.csv`
- `data/p54.csv`
- `data/p69.csv`

Open file `collect_data.py` and check these lines before running the command: `python collect_data.py`

```
file_name = 'p54' # predicate property
file_name_output = 'data/output_' + file_name + '.csv'  # the output url
page_list = read_from_csv_file('data/' + file_name + '.csv',  ',', 1) # the input url
page_list = sorted(list(set(page_list))) # filter repetitive pages and sort by alphabet
```
**The data collection process will take time about several days to a week, consider to hangout your code on the server.**

# How to evaluate your mapped data?

To check the mapped data, we use several methods:
- Entity linking methods
- Cumulative rates
- Noise fitering
- Relationships between sentence predicates against Wikidata properties and qualifiers

**Entity linking methods**

# Contact
You can contact me by email: tahoangthang@gmail.com
