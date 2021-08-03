#........................................................................................................
# Title: Wikidata claims (statements) to natural language (a part of Triple2Text/Ontology2Text task)
# Author: Ta, Hoang Thang
# Email: tahoangthang@gmail.com
# Lab: https://www.cic.ipn.mx
# Date: 12/2019
#........................................................................................................

from corpus_estimation import *
from base import *
from autoencoders import *

# basic statistics
def test_statistics():
  
    #basic_statistics('our_data/output_common.csv', '#', dtypes, True)
    basic_statistics('our_data/output_common2.csv', '#', dtypes, True)
    basic_statistics('our_data/output_p26.csv', '#', dtypes, True)
    basic_statistics('our_data/output_p39.csv', '#', dtypes, True)
    basic_statistics('our_data/output_p54.csv', '#', dtypes, True)
    basic_statistics('our_data/output_p69.csv', '#', dtypes, True)
    basic_statistics('our_data/output_p108.csv', '#', dtypes, True)
    basic_statistics('our_data/output_p166.csv', '#', dtypes, True)

# test cumulative_rate
def test_cumulative_rate(df):

    total_cumulative_list = []
    labels = ['p26', 'p39', 'p54', 'p69', 'p108', 'p166', 'common']
    
    list1  = cumulative_rate_by_property('p26', df)
    list2  = cumulative_rate_by_property('p39', df)
    list3  = cumulative_rate_by_property('p54', df)
    list4  = cumulative_rate_by_property('p69', df)
    list5  = cumulative_rate_by_property('p108', df)
    list6  = cumulative_rate_by_property('p166', df)
    list7  = cumulative_rate_by_property('common', df)

    total_cumulative_list.append(list1)
    total_cumulative_list.append(list2)
    total_cumulative_list.append(list3)
    total_cumulative_list.append(list4)
    total_cumulative_list.append(list5)
    total_cumulative_list.append(list6)
    total_cumulative_list.append(list7)
    
    sentence_plot_by_redundant_words(total_cumulative_list, labels, 'Cumulative rate of sentences by redundant words',
                                     'Number of redundant words', 'Cumulative rate of sentences')


# sentence plot by distance
def test_filter_noise_by_cumulative_rate(property_name, rate=0, top_words=0):

    file_name, sen_list, best_sentences, labeled_sen_list, counter, idf_dict, word_corpus, model = load_corpus(property_name)
    sentences, number_redundant_word_list, redundant_word_list = get_corpus_redundant_words(sen_list)
    rank_list = rank_sentence_by_redundant_words(number_redundant_word_list)
    cumulative_list = cumulative_rate(rank_list)
    sentence_plot_by_redundant_words(cumulative_list, 'Cumulative rate of sentences by redundant words', 'Number of redudant words', 'Cumulative rate of sentences')

    # filter noise by cumulative_rate
    sentences, redundant_word_list, number_redundant_word_list = filter_noise_by_cumulative_rate(sentences,
                                redundant_word_list, number_redundant_word_list, cumulative_list, rate, top_words)

    '''for x, y in zip(sentences, number_redundant_word_list):
        print(x, y)'''
    return sentences, redundant_word_list, number_redundant_word_list


def test_convert_corpus_to_measures(result_dict, output_file):
    #print('result_dict: ', result_dict)

    """
        Input: result_dict
        Output: output_file
    """

    convert_corpus_to_measures(output_file, result_dict['sen_list'], result_dict['best_sentences'],
                           result_dict['local_model'], result_dict['global_model'], result_dict['counter'],
                           result_dict['idf_dict'])
    

def test_noise_filtering(x_train, x_test, x_true, y_train, y_test, y_all, file_name = 'result.csv'):

    result_list = []

    
    write_to_new_text_file(file_name, 'method,executed_time,noise_rate,fm,uca,purity,silhouette')

    #...............................................
    # before training  
    dict_list = compare_clustering_methods(x_true, y_true, ['nearestneighbors01', 'nearestneighbors02', 'dbscan01', 'dbscan02',
                                                            'dbscan03', 'dbscan04', 'optics01', 'agglomerative', 'localoutliner01',
                                                            'localoutliner02', 'localoutliner03', 'localoutliner04', 'birch01',
                                                            'birch02', 'kmeans'], 'pca')
      
    print('results: ', dict_list)
    result_list.append(dict_list)
    show_all_plots(dict_list)

    for item in dict_list:
        criteria_dict = item['criteria_dict']
        temp_list = [ 'cm1-' + item['method'], item['executed_time'], criteria_dict['noise_rate'], criteria_dict['fowlkes_mallows'],
                     criteria_dict['unsupervised_clustering_accuracy'], criteria_dict['purity'], criteria_dict['silhouette']]
        data = ','.join(str(x) for x in temp_list)
        write_to_text_file(file_name, data)


    dict_list = compare_clustering_methods(x_true, y_true, ['nearestneighbors01', 'nearestneighbors02', 'dbscan01', 'dbscan02',
                                                            'dbscan03', 'dbscan04', 'optics01', 'agglomerative', 'localoutliner01',
                                                            'localoutliner02', 'localoutliner03', 'localoutliner04', 'birch01',
                                                            'birch02', 'kmeans'], 'pca', dr_first=True)
        

    print('results: ', dict_list)
    result_list.append(dict_list)
    show_all_plots(dict_list)

    for item in dict_list:
        criteria_dict = item['criteria_dict']
        temp_list = [ 'cm2-' + item['method'], item['executed_time'], criteria_dict['noise_rate'], criteria_dict['fowlkes_mallows'],
                     criteria_dict['unsupervised_clustering_accuracy'], criteria_dict['purity'], criteria_dict['silhouette']]
        data = ','.join(str(x) for x in temp_list)
        write_to_text_file(file_name, data)

    # after training
    # vanilla autoencoder
    history_list, total_loss_list, total_val_loss_list, hidden_layer_data = train_autoencoder(4, 2, x_train, x_test, x_true, y_train,
                y_test, y_true, 500, 2, 16659, centroids = [], loss_coefficient=0.5, cluster_loss_type='euclidean',
                network_loss='mean_squared_error', optimizer='sgd', exit_type='total_val',
                min_delta=0.001)

    show_loss_plot(history_list, total_loss_list, total_val_loss_list)

    
    dict_list = compare_clustering_methods(hidden_layer_data, y_true, ['nearestneighbors01', 'nearestneighbors02', 'dbscan01', 'dbscan02',
                                                            'dbscan03', 'dbscan04', 'optics01', 'agglomerative', 'localoutliner01',
                                                            'localoutliner02', 'localoutliner03', 'localoutliner04', 'birch01',
                                                            'birch02', 'kmeans'], 'pca')
    
    print('results: ', dict_list)
    result_list.append(dict_list)
    show_all_plots(dict_list)

    for item in dict_list:
        criteria_dict = item['criteria_dict']
        temp_list = [ 'cm3-' + item['method'], item['executed_time'], criteria_dict['noise_rate'], criteria_dict['fowlkes_mallows'],
                     criteria_dict['unsupervised_clustering_accuracy'], criteria_dict['purity'], criteria_dict['silhouette']]
        data = ','.join(str(x) for x in temp_list)
        write_to_text_file(file_name, data)

    #...............................................
    # denoising autoencoder
    history_list, total_loss_list, total_val_loss_list, hidden_layer_data = train_autoencoder(4, 2, x_train, x_test, x_true, y_train,
                y_test, y_true, 500, 2, 16659, centroids = [], loss_coefficient=0.5, cluster_loss_type='euclidean',
                network_loss='mean_squared_error', optimizer='sgd', exit_type='total_val',
                min_delta=0.001, denoising=True, noise_factor=1)

    show_loss_plot(history_list, total_loss_list, total_val_loss_list)
    
    # after training
    dict_list = compare_clustering_methods(hidden_layer_data, y_true, ['nearestneighbors01', 'nearestneighbors02', 'dbscan01', 'dbscan02',
                                                            'dbscan03', 'dbscan04', 'optics01', 'agglomerative', 'localoutliner01',
                                                            'localoutliner02', 'localoutliner03', 'localoutliner04', 'birch01',
                                                            'birch02', 'kmeans'], 'pca')
    
    print('results: ', dict_list)
    result_list.append(dict_list)
    show_all_plots(dict_list)

    for item in dict_list:
        criteria_dict = item['criteria_dict']
        temp_list = [ 'cm4-' + item['method'], item['executed_time'], criteria_dict['noise_rate'], criteria_dict['fowlkes_mallows'],
                     criteria_dict['unsupervised_clustering_accuracy'], criteria_dict['purity'], criteria_dict['silhouette']]
        data = ','.join(str(x) for x in temp_list)
        write_to_text_file(file_name, data)

    return result_list

def test_rank_predicate_by_property_and_qualifier(by_qualifier=False):
    predicate_list = []

    result_dict = load_corpus('output_p26.csv', 'wordvectors_p26.txt', 'p26', '#', dtypes, False, False)
    predicate_dict = rank_predicate(result_dict['sen_list'], result_dict['best_sentences'], result_dict['counter'],
                                             result_dict['local_model'], result_dict['global_model'], by_qualifier)
    predicate_list.append(predicate_dict)
    group_dict = group_predicate(predicate_dict, 10, True)
  

    result_dict = load_corpus('output_p39.csv', 'wordvectors_p39.txt', 'p39', '#', dtypes, False, False)
    predicate_dict = rank_predicate(result_dict['sen_list'], result_dict['best_sentences'], result_dict['counter'],
                                             result_dict['local_model'], result_dict['global_model'], by_qualifier)
    predicate_list.append(predicate_dict)
    group_dict = group_predicate(predicate_dict, 10, True)

    
    result_dict = load_corpus('output_p54.csv', 'wordvectors_p54.txt', 'p54', '#', dtypes, False, False)
    predicate_dict = rank_predicate(result_dict['sen_list'], result_dict['best_sentences'], result_dict['counter'],
                                             result_dict['local_model'], result_dict['global_model'], by_qualifier)
    predicate_list.append(predicate_dict)
    group_dict = group_predicate(predicate_dict, 10, True)


    result_dict = load_corpus('output_p69.csv', 'wordvectors_p69.txt', 'p69', '#', dtypes, False, False)
    predicate_dict = rank_predicate(result_dict['sen_list'], result_dict['best_sentences'], result_dict['counter'],
                                             result_dict['local_model'], result_dict['global_model'], by_qualifier)
    predicate_list.append(predicate_dict)
    group_dict = group_predicate(predicate_dict, 10, True)


    result_dict = load_corpus('output_p108.csv', 'wordvectors_p108.txt', 'p108', '#', dtypes, False, False)
    predicate_dict = rank_predicate(result_dict['sen_list'], result_dict['best_sentences'], result_dict['counter'],
                                             result_dict['local_model'], result_dict['global_model'], by_qualifier)
    predicate_list.append(predicate_dict)
    group_dict = group_predicate(predicate_dict, 10, True)
    

    result_dict = load_corpus('output_p166.csv', 'wordvectors_p166.txt', 'p166', '#', dtypes, False, False)
    predicate_dict = rank_predicate(result_dict['sen_list'], result_dict['best_sentences'], result_dict['counter'],
                                             result_dict['local_model'], result_dict['global_model'], by_qualifier)
    predicate_list.append(predicate_dict)
    group_dict = group_predicate(predicate_dict, 10, True)
    
    show_predicate_plot(predicate_list, ['P26', 'P39', 'P54', 'P69', 'P108', 'P166'])
    #show_predicate_plot(predicate_list, ['P26'])

#........................................................................................................
# -----------------------------------
# add metrics (IDF, TF, local & global distances) to the corpus, formulas (26), (27), (28), (29)
#result_dict = load_corpus('data/output_common2.csv', 'data/wordvectors_common2.txt', 'common2', '#', dtypes, False, True)
#test_convert_corpus_to_measures(result_dict, 'data/output_common2_measures.csv')
# -----------------------------------

# -----------------------------------
# basic statistics # (Section 6.2 & Table 11)
# test_statistics()
# -----------------------------------

# -----------------------------------
# noise filtering ---------------------
'''input_file_name = 'data/output_common2_measures.csv'   
df = pd.read_csv(input_file_name, delimiter='#', dtype=dtypes3, usecols=list(dtypes3))
df = df.sample(frac=1.0)'''
#df_elements = df.sample(n=100, random_state=0) # 100 random rows

# cumulative rate
#test_cumulative_rate(df) # Figure 6

# noise filtering - Section 6.3 & Table 12
'''label_list = df['label'].tolist()
y_true = []
for la in label_list:
    if (la == 'x'): y_true.append(0)
    else: y_true.append(-1) # outliners

x_true = []
df1 = df.loc[:, ['tf2', 'idf2', 'local2', 'global2']]
cols = df1.columns
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
x_true = [list(row[1:]) for row in df[cols].itertuples(name=None)]

x_true = np.array(np.float32(x_true)) # to avoid buffer overflow
x_true = MinMaxScaler(feature_range=(0, 100)).fit(x_true).transform(x_true) # fit range [0, 100]

y_true = np.array(y_true)
x_train, x_test, y_train, y_test = train_test_split(x_true, y_true, test_size=0.1, random_state=1)

result_list = test_noise_filtering(x_train, x_test, x_true, y_train, y_test, y_true)
print('result_list: ', result_list)'''
# -----------------------------------

# -----------------------------------
# rank qualifiers by predicates - Table 14 & Table 15
#test_rank_predicate_by_property_and_qualifier(by_qualifier=False) # Table 14
#test_rank_predicate_by_property_and_qualifier(by_qualifier=True)  # Table 15
# -----------------------------------
#........................................................................................................
