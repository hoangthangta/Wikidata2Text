from read_write_file import *

import pandas as pd
import numpy as np
import math

# untrained
dtypes = {
    'type': 'category',
    'subject': 'category',
    'predicate': 'category',
    'object': 'category',
    'qualifiers': 'category',
    'raw_sentence': 'category',
    'subject_value': 'category',
    'subject_matching': 'category',
    'root': 'category',
    'prepositional_verb': 'category',
    'verbs': 'category',
    'predicate_matching': 'category',
    'object_value': 'category',
    'object_matching': 'category',
    'qualifier_value': 'category',
    'qualifier_matching': 'category',
    'labeled_sentence_1': 'category',
    'labeled_sentence_2': 'category',
    'order_1': 'category',
    'order_2': 'category',
    }

# trained    
dtypes2 = {
    'type': 'category',
    'subject': 'category',
    'predicate': 'category',
    'object': 'category',
    'qualifiers': 'category',
    'raw_sentence': 'category',
    'subject_value': 'category',
    'subject_matching': 'category',
    'root': 'category',
    'prepositional_verb': 'category',
    'verbs': 'category',
    'predicate_matching': 'category',
    'object_value': 'category',
    'object_matching': 'category',
    'qualifier_value': 'category',
    'qualifier_matching': 'category',
    'labeled_sentence_1': 'category',
    'labeled_sentence_2': 'category',
    'order_1': 'category',
    'order_2': 'category',
    'trained_sentence': 'category',
    }

# measures 
dtypes3 = {
    'type': 'category',
    'subject': 'category',
    'predicate': 'category',
    'object': 'category',
    'qualifiers': 'category',
    'raw_sentence': 'category',
    'subject_value': 'category',
    'subject_matching': 'category',
    'root': 'category',
    'prepositional_verb': 'category',
    'verbs': 'category',
    'predicate_matching': 'category',
    'object_value': 'category',
    'object_matching': 'category',
    'qualifier_value': 'category',
    'qualifier_matching': 'category',
    'labeled_sentence_1': 'category',
    'labeled_sentence_2': 'category',
    'order_1': 'category',
    'order_2': 'category',
    'label': 'category',
    'redundant_words': 'category',
    'length': 'category',
    'tf1': 'category',
    'tf2': 'category',
    'idf1': 'category',
    'idf2': 'category',
    'local1': 'category',
    'local2': 'category',
    'global1': 'category',
    'global2': 'category',
    'tf_idf1': 'category',
    'tf_idf2': 'category',
    'local_tf1': 'category',
    'local_tf2': 'category',
    'local_idf1': 'category',
    'local_idf2': 'category',
    'local_tf_idf1': 'category',
    'local_tf_idf2': 'category',
    'global_tf1': 'category',
    'global_tf2': 'category',
    'global_idf1': 'category',
    'global_idf2': 'category',
    'global_tf_idf1': 'category',
    'global_tf_idf2': 'category',
    'global_qualifier1': 'category',
    'global_qualifier2': 'category',
    'global_qualifier_tf1': 'category',
    'global_qualifier_tf2': 'category',
    'global_qualifier_idf1': 'category',
    'global_qualifier_idf2': 'category',
    'global_qualifier_tf_idf1': 'category',
    'global_qualifier_tf_idf2': 'category',
    }