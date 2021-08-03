# Wikidata2Text

Wikidata2Text (Wikidata statements to Text - WS2T) is a sub-task of Data2Text. This task is to translate Wikidata claims or statements, organized as a set of triples or quadruples to Wikipedia sentences.  

Our paper are currently under review at http://www.semantic-web-journal.net under the name "`Mapping Process for the Task: Wikidata Statements to Text as Wikipedia Sentences`". The code is mainly about the data mapping process.

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

To get the trained text, we use 2 files: `neuralcoref_training.py` and `test_neuralcoref.py`. Here is the content of `test_neuralcoref.py`.

```
from neuralcoref_training import *
# output will be stored at "data/output_common2_trained.csv"
input_file_name = 'data/output_common2.csv' # input
df = pd.read_csv(input_file_name, delimiter='#', dtype=dtypes, usecols=list(dtypes))
get_trained_sentence(df)
```

We already evaluated the corpus by these EL methods which stored in folder "`/our_data`". For example, check one of these files: `our_data/subject_matching_trained_wat.txt`. All of these files are in *.csv format with the form:

For type matching:
```sentence, total_score, current_rows, no_datatype_rows, no_qualifier_rows, sentence_score, sentence_length```

For data matching:
```sentence, total_score, current_rows, match_rows, no_term_rows, current_rows - match_rows - no_term_rows```

- total_score: This is the total score for the corpus. For each sentence, the maximum score is 1.
- no_datatype_rows: the number of rows which do not have datatype for `type matching`.
- no_qualifier_rows: the number of rows which do not have any qualifier. All sentences in our corpus contains at least 1 qualifier.
- match_rows: is the number of rows that match the data.
- no_term_rows: is the number of rows that entity linking methods can not get the data.

**2. Evaluation all sentences by metrics (TF, IDF, local_distance, global_distance and their combinations)**

Open `test_corpus_estimation.py` and see these lines:

```
result_dict = load_corpus('data/output_common2.csv', 'data/wordvectors_common2.txt', 'common2', '#', dtypes, False, True)
test_convert_corpus_to_measures(result_dict, 'data/output_common2_measures.csv')
```

Note that we use "`D:\wiki-news-300d-1M.vec`" in the function `load_corpus()` in the file `corpus_estimation.py`. Download `wiki-news-300d-1M.vec` at https://fasttext.cc/docs/en/english-vectors.html.

**We already did this step and store as the file "output_common2_measures.csv" in "/data" and "/our_data" folders.**
Note that the output file `output_common2_measures.csv` will be used to evaluate for all below sections.

**3. Basic statistics**

Open `test_corpus_estimation.py` and see these lines:

```
basic statistics # (Section 6.2 & Table 11)
test_statistics()
```

**4. Cumulative rate**

![alt text](https://github.com/tahoangthang/Wikidata2Text/blob/main/sentence_plot_by_redundant_words.svg?raw=true)

By each Wikidata property, we extract redundant words of their sentences and show them on the plot. Open `test_corpus_estimation.py` and see these lines:

```
input_file_name = 'data/output_common2_measures.csv'   
df = pd.read_csv(input_file_name, delimiter='#', dtype=dtypes3, usecols=list(dtypes3))
df = df.sample(frac=1.0)
test_cumulative_rate(df) # Figure 6
```

**5. Noise filtering**

Open `test_corpus_estimation.py` and see these lines:

```
input_file_name = 'data/output_common2_measures.csv'   
df = pd.read_csv(input_file_name, delimiter='#', dtype=dtypes3, usecols=list(dtypes3))
df = df.sample(frac=1.0)

# noise filtering - Section 6.3 & Table 12
label_list = df['label'].tolist()
y_true = []
for la in label_list:
    if (la == 'x'): y_true.append(0)
    else: y_true.append(-1) # outliners
x_true = []
df1 = df.loc[:, ['tf2', 'idf2', 'local2', 'global2']] # we use 4 features
cols = df1.columns
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
x_true = [list(row[1:]) for row in df[cols].itertuples(name=None)]
x_true = np.array(np.float32(x_true)) # to avoid buffer overflow
x_true = MinMaxScaler(feature_range=(0, 100)).fit(x_true).transform(x_true) # fit range [0, 100]
y_true = np.array(y_true)
x_train, x_test, y_train, y_test = train_test_split(x_true, y_true, test_size=0.1, random_state=1)
result_list = test_noise_filtering(x_train, x_test, x_true, y_train, y_test, y_true)
print('result_list: ', result_list)
```

We store the result as `our_data/result_noise_filtering.csv`.

**6. Relationships between sentence predicates against Wikidata properties and qualifiers**

Open `test_corpus_estimation.py` and see these lines:

```
# rank qualifiers by predicates - Table 14 & Table 15
test_rank_predicate_by_property_and_qualifier(by_qualifier=False) # Table 14, without qualifiers
test_rank_predicate_by_property_and_qualifier(by_qualifier=True)  # Table 15, with qualifiers
```

Note that we use "`D:\wiki-news-300d-1M.vec`" in the function `load_corpus()` in the file `corpus_estimation.py`. Download `wiki-news-300d-1M.vec` at https://fasttext.cc/docs/en/english-vectors.html.

We store the results as `/our_data/results_roots_vs_properties2.txt` and `our_data/results_roots_vs_properties_and_qualifiers.txt`.

![alt text](https://github.com/tahoangthang/Wikidata2Text/blob/main/relationships_predicates_properties.png?raw=true)
Relationships between sentence predicates and Wikidata properties.

# Contact
You can contact me by email: 
- tahoangthang@gmail.com
- thangth@dlu.edu.vn
