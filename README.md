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

**1. Entity linking methods (EL methods)**

We use several entity linking methods: AIDA, Babel (optional), OpenTapioca, TagMe, WAT, Wikifer, and our baseline method mapping terms directly to Wikipedia and Wikidata. These methods can be found in the folder `/entity_link`. The main code is in the file `mapping_estimation.py`.

We evaluate the mapped results by two types: `type matching`, and `data matching`, over three sentence components: `subject matching`, `object matching`, and `qualifier matching` by `raw text` and `trained text` (neuralcoref). We only apply `trained text` for `subject matching` since there are a lot of pronouns (he, she) in the mapped sentences.

You can find these files:
- test_object_matching_aida.py
- test_object_matching_babelfy.py
- test_object_matching_opentapioca.py
- ...

For example, open the file `test_object_matching_aida.py`:
```
from mapping_estimation import *

input_file_name = 'data/output_common2.csv' # the input data
df = pd.read_csv(input_file_name, delimiter='#', dtype=dtypes, usecols=list(dtypes))

evaluate_object_matching('aida', df) # the output will be store on "/data" folder
```

We already evaluated the corpus by these EL methods which stored in folder "`/our_data`". All of these files are in *.csv format with the form:

For type matching:
```sentence, total_score, current_rows, no_datatype_rows, no_qualifier_rows, sentence_score, sentence_length```

For data matching:
```sentence, total_score, current_rows, match_rows, no_term_rows, current_rows - match_rows - no_term_rows```

- total_score: This is the total score for the corpus. For each sentence, the maximum score is 1.
- no_datatype_rows: the number of rows which do not have datatype for `type matching`.
- no_qualifier_rows: the number of rows which do not have any qualifier. All sentences in our corpus contains at least 1 qualifier.
- match_rows: is the number of rows that match the data.
- no_term_rows: is the number of rows that entity linking methods can not get the data.


# Contact
You can contact me by email: 
- tahoangthang@gmail.com
- thangth@dlu.edu.vn
