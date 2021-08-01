# References:
#   1. https://hdbscan.readthedocs.io/en/latest/performance_and_scalability.html

from cluster_losses import *
from cluster_metrics import *
from corpus_estimation import *

from sklearn.decomposition import TruncatedSVD, PCA, NMF, SparsePCA, FastICA
from sklearn.preprocessing import MinMaxScaler

def show_clustering_criteria(data, idx, y_all):
    """
        y_all = y_true
    """
    id_dict = {}
    for id in idx:
        if (id not in id_dict): id_dict[id] = 1
        else: id_dict[id] += 1

    #print('id_dict: ', id_dict)
    max_ = max([v for k, v in id_dict.items()])

    result_dict = {}

    result_dict['n_noises'] = len(idx) - max_
    result_dict['noise_rate'] = 1 - (max_/len(idx))

    #result_dict['rand_index'] = cluster_ri(idx, y_all)
    result_dict['homogeneity'] = cluster_homogeneity(idx, y_all)
    result_dict['completeness'] = cluster_completeness(idx, y_all)
    result_dict['v_measure'] = cluster_v_measure(idx, y_all)
    result_dict['adjusted_rand_index'] = cluster_ari(idx, y_all)
    result_dict['adjusted_mutual_info'] = cluster_ami(idx, y_all)
    result_dict['normalized_mutual_info'] = cluster_nmi(idx, y_all)
    result_dict['fowlkes_mallows'] = cluster_fm(idx, y_all)
    result_dict['unsupervised_clustering_accuracy'] = cluster_acc(idx, y_all)
    result_dict['purity'] = cluster_purity(idx, y_all)
    result_dict['silhouette'] = cluster_silhouette(data, idx, metric='euclidean')

    return result_dict

def clustering_results(data, y_all, method='kmeans', dr_method='pca', dr_first=False):

    # dimensionality reduction first
    if (dr_first == True):
        if (len(data[0]) > 2): # if the number of dimensions > 2
            data = decomposition(data, dr_method, 2)

    # calculate the executed time
    start_time = time.time()
    #print('data, method: ', data, method)   
    clustering, idx = clustering_by_methods(data, method)
    end_time = time.time() - start_time
    
    criteria_dict = show_clustering_criteria(data, idx, y_all)

    # for showing only 
    if (dr_first == False):
        data = decomposition(data, dr_method, 2)
    data = MinMaxScaler().fit(data).transform(data)

    return {'method': method, 'executed_time': end_time, 'dr_method': dr_method, 'data': data, 'idx': idx,
            'criteria_dict': criteria_dict}

def compare_clustering_methods(x_all, y_all, method_list, dr_method, dr_first=False):
    """
        x_all: data
        y_all: y_true or labels
        method_list: a list of clustering methods used to compare
        dr_method: dimensionality reduction method (PCA, NMF, etc)
    """

    dict_list = []
    for m in method_list:
        dict_list.append(clustering_results(x_all, y_all, method=m, dr_method=dr_method, dr_first=dr_first))
    
    return dict_list


def show_all_plots(dict_list):

    columns = 3
    temp_index = int(len(dict_list)/columns)
    remainer = len(dict_list)%columns

    if (remainer != 0): temp_index += 1
    
    fig, axs = plt.subplots(temp_index, columns)
    plt.rcParams.update({'font.size': 8})

    #plt.rcParams.update({'axes.titlesize': 'medium'})

    for i in range(0, temp_index):
        for j in range(0, columns):
            try:
                # use same colors for all plots
                #colors = dict_list[i*2 + j]['idx']
                colors = normalize_labels(dict_list[i*columns + j]['idx'])
 
                axs[i, j].scatter(dict_list[i*columns + j]['data'][:,0], dict_list[i*columns + j]['data'][:,1], c = colors, s = 10)
                axs[i, j].set_title(dict_list[i*columns + j]['method'], fontsize=8)

                #criteria_dict = dict_list[i*2 + j]['criteria_dict']
                #temp_string = ''
                #temp_string += 'Noise rate: ' + str(round(criteria_dict['noise_rate'], 4)) + '\n'
                #temp_string += 'Purity: ' + str(round(criteria_dict['purity'], 4)) + '\n'
                #temp_string += 'Silhouette: ' + str(round(criteria_dict['silhouette'], 4)) + '\n'
                #axs[i, j].text(0, 0, temp_string)
            except: 
                continue 

    for ax in axs.flat:
        ax.set(xlabel='x', ylabel='y')
    
    for ax in axs.flat:
        ax.label_outer()

    plt.show()
        
def normalize_labels(labels):
    min_value = min(labels)
    #max_value = max(labels)
    
    if (min_value < 0):
        labels = labels + abs(min(labels)) # to avoid negative labels
        
    return labels

def clustering_by_methods(data, method='kmeans'):

    """
        method: kmeans (default)
    """
    clustering = []
    idx = []
    if (method == 'dbscan01'):
        clustering = DBSCAN(eps=3.25, min_samples=5).fit(data)
    elif (method == 'dbscan02'):
        clustering = DBSCAN(eps=5.25, min_samples=5).fit(data)
    elif (method == 'dbscan03'):
        clustering = DBSCAN(eps=7.25, min_samples=5).fit(data)
    elif (method == 'dbscan04'):
        clustering = DBSCAN(eps=9.25, min_samples=5).fit(data)
    elif (method == 'optics01'):
        clustering = OPTICS(min_cluster_size=int(len(data)*0.5)).fit(data)
    elif (method == 'optics02'):
        clustering = OPTICS(min_cluster_size=int(len(data)*0.05)).fit(data)
    elif (method == 'optics03'):
        clustering = OPTICS(min_cluster_size=int(len(data)*0.001)).fit(data) 
    elif (method == 'meanshift'): # take a very long time
        clustering = MeanShift().fit(data)
    elif (method == 'affinity'): # take a very long time
        clustering = AffinityPropagation().fit(data)
    elif (method == 'spectralclustering'): # take a very long time
        clustering = SpectralClustering(n_clusters=2, assign_labels='discretize').fit(data)
    elif (method == 'agglomerative'): 
        clustering = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete').fit(data)
    elif (method == 'localoutliner01'):
        clustering = LocalOutlierFactor(n_neighbors=int(len(data)*0.1), contamination=0.01).fit_predict(data) 
    elif (method == 'localoutliner02'):
        clustering = LocalOutlierFactor(n_neighbors=int(len(data)*0.1), contamination=0.005).fit_predict(data)
    elif (method == 'localoutliner03'):
        clustering = LocalOutlierFactor(n_neighbors=int(len(data)*0.1), contamination=0.001).fit_predict(data)
    elif (method == 'localoutliner04'):
        clustering = LocalOutlierFactor(n_neighbors=int(len(data)*0.1), contamination=0.0005).fit_predict(data)
    elif (method == 'birch01'):
        clustering = Birch(n_clusters=2, threshold=0.5).fit(data)
    elif (method == 'birch02'):
        clustering = Birch(n_clusters=2, threshold=0.1).fit(data)
    elif (method == 'birch03'):
        clustering = Birch(n_clusters=2, threshold=0.07).fit(data)
    elif (method == 'birch04'):
        clustering = Birch(n_clusters=2, threshold=0.04).fit(data)
    elif (method == 'gaussianmixture01'):
        clustering = GaussianMixture(n_components=2, covariance_type='diag').fit(data)       
    elif (method == 'gaussianmixture02'):
        clustering = GaussianMixture(n_components=2, covariance_type='spherical').fit(data)
    elif (method == 'gaussianmixture03'):
        clustering = GaussianMixture(n_components=2, covariance_type='tied').fit(data)
    elif (method == 'gaussianmixture04'):
        clustering = GaussianMixture(n_components=2, covariance_type='full').fit(data)
    elif ('nearestneighbors' in method):
        outlier_indexes = []
        distances, indexes = NearestNeighbors(n_neighbors=5).fit(data).kneighbors(data)
        if (method == 'nearestneighbors01'):
            outlier_indexes = np.where(distances.mean(axis = 1) > 4)[0]
        elif(method == 'nearestneighbors02'):
            outlier_indexes = np.where(distances.mean(axis = 1) > 6)[0]
        #print('total mean: ', distances.mean(axis=1).mean()) # 0.1325
        #plt.plot(distances.mean(axis=1))
        #plt.show()
        idx = []
        for k, v in enumerate(data):
            if (k in outlier_indexes): idx.append(-1)
            else: idx.append(0)
        idx = np.array(idx)
    else:
        clustering = KMeans(n_clusters=2).fit(data) 

    if ('gaussianmixture' in method):
        idx = clustering.predict(data)
    elif ('localoutliner' in method):
        idx = clustering
    elif ('nearestneighbors' not in method):   
        idx = clustering.labels_
    
    return clustering, idx

# decomposition: convert high dimensions to low dimensions
def decomposition(matrix, decomposition_type, dimension = 2):

    # t-SNE will be applied in the future
    
    if (decomposition_type == 'truncatedsvd'):
        return TruncatedSVD(n_components=dimension).fit_transform(matrix)
    elif(decomposition_type == 'pca'):
        return PCA(n_components=dimension).fit_transform(matrix)
    elif(decomposition_type == 'sparsepca'):
        return SparsePCA(n_components=dimension, random_state=0).fit_transform(matrix)
    elif(decomposition_type == 'nmf'): 
        matrix = MinMaxScaler().fit(matrix).transform(matrix) # scale to [0-1] to avoid negative values
        return NMF(n_components=dimension, init='random', random_state=0).fit_transform(matrix)
    elif(decomposition_type == 'fastica'):
        return FastICA(n_components=dimension,random_state=0).fit_transform(matrix)
    return matrix
