from cluster_losses import *
from cluster_metrics import *
from corpus_estimation import *
from cluster_methods import *

from sklearn.model_selection import train_test_split

import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

def add_noise(x_train, x_test, noise_factor):
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.5, scale=0.5, size=x_train.shape) 
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.5, scale=0.5, size=x_test.shape) 

    x_train_noisy = np.clip(x_train_noisy, 0., 10.)
    x_test_noisy = np.clip(x_test_noisy, 0., 10.)

    print('x_test_noisy: ', x_test_noisy)
    return x_train_noisy, x_test_noisy

def calculate_total_loss(network_loss, cluster_loss, loss_coefficient):
    return network_loss*loss_coefficient + (1-loss_coefficient)*cluster_loss

def train_autoencoder(input_dim, encoding_dim, x_train, x_test, x_all, y_train, y_test, y_all, epochs, n_clusters, batch_size=1,
                      centroids=[], loss_coefficient=0.5, cluster_loss_type='euclidean', network_loss='mean_squared_error',
                      optimizer='sgd', exit_type='total_val', min_delta=0.001, denoising=False, noise_factor=0.5):

    """
        x_all = x_true
        y_all = y_true
    """
    # denoising autoencoder
    x_train_noisy, x_test_noisy = [], []
    if (denoising == True):
        x_train_noisy, x_test_noisy = add_noise(x_train, x_test, noise_factor)
    
    input_data = Input(shape=(input_dim,)) # input_dim = 32

    encoded = Dense(encoding_dim)(input_data)  
    
    # set up more hidden layers here
    decoded = Dense(input_dim)(encoded) # "decoded" is the lossy reconstruction of the input
    
    autoencoder = Model(input_data, decoded) # this model maps an input to its reconstruction

    encoder = Model(input_data, encoded) # this model maps an input to its encoded representation
    encoded_input = Input(shape=(encoding_dim,)) # create a placeholder for an encoded (2-dimensional) input
    
    decoder_layer = autoencoder.layers[-1] # retrieve the last layer of the autoencoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input)) # create the decoder model

    hidden_layer = autoencoder.layers[0] # get hidden layer values
    hidden_layer_decoder = Model(encoded_input, hidden_layer(encoded_input))

    autoencoder.compile(loss=network_loss, optimizer=optimizer)

    total_loss_list = [] # total lost
    total_val_loss_list = [] # total validation lost
    
    hidden_layer_data = []
    hidden_layer_data_list = []
    decoded_data = []
    
    centroids = []
    cluster_loss = 0
    
    history_list = []
    idx = []
    for ep in range(epochs):

        print('..................................................')
        print('epoch: \t', ep)

        if (denoising == True):
            history = autoencoder.fit(x_train_noisy, x_train,
                                  batch_size=batch_size, epochs=1,
                                  shuffle=True,
                                  validation_data=(x_test_noisy, x_test),
                                  verbose=0)
        else:
            history = autoencoder.fit(x_train, x_train,
                                  batch_size=batch_size, epochs=1,
                                  shuffle=True,
                                  validation_data=(x_test, x_test),
                                  verbose=0)
        #callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

        history_list.append(history) # add trained values to history in every epoch

        encoded_data = encoder.predict(x_all) # get all data
        #print('encoded_data: ', encoded_data)
        
        decoded_data = decoder.predict(encoded_data) 
        #print('decoded_data: ', decoded_data)
        
        hidden_layer_data = hidden_layer_decoder.predict(encoded_data)
        #print('hidden_layer_data: ', hidden_layer_data)
        hidden_layer_data_list.append(hidden_layer_data)

        # calculate loss
        network_loss = autoencoder.history.history['loss'][0]
        network_val_loss = autoencoder.history.history['val_loss'][0]

        print('network_loss, network_val_loss: ', network_loss, network_val_loss)

        centroids = []
        for x, y in zip(hidden_layer_data, y_all):
            if (y == 0): # true labels
                centroids.append(x.tolist())
        #print('centroids: ', centroids, len(centroids))

        cluster_sum_loss, cluster_mean_loss = 0, 0

        if (cluster_loss_type == 'kmeans'):
            cluster_sum_loss, cluster_mean_loss, centroids, idx = kmeans_loss(hidden_layer_data, n_clusters)
        elif (cluster_loss_type == 'euclidean'):
            # need to have centroids
            cluster_sum_loss, cluster_mean_loss = euclidean_distance_mean_loss(hidden_layer_data, centroids)
        elif(cluster_loss_type == 'rmse'):
            # need to have centroids
            cluster_sum_loss, cluster_mean_loss = rmse_loss(hidden_layer_data, centroids)
        else:
            # similar to kmeans but need to have centroids
            cluster_sum_loss, cluster_mean_loss = mse_loss(hidden_layer_data, centroids)
            print('cluster_sum_loss, cluster_mean_loss: ', cluster_sum_loss, cluster_mean_loss)
                
        total_loss = calculate_total_loss(network_loss, cluster_mean_loss, loss_coefficient)
        total_loss_list.append(total_loss)

        total_val_loss = calculate_total_loss(network_val_loss, cluster_mean_loss, loss_coefficient)
        total_val_loss_list.append(total_val_loss)

        print('total loss, total validation loss: ', total_loss, total_val_loss)

        #if (ep <= 5): continue

        # break condition

        if (exit_type == 'total'):
            condition = exit_condition(total_loss_list, min_delta)
            if (condition != -1):
                if (condition == 0):
                    return history_list[:-1], total_loss_list[:-1], total_val_loss_list[:-1], hidden_layer_data_list[-2]
                break
        elif (exit_type == 'total_val'):
            condition = exit_condition(total_val_loss_list, min_delta)
            if (condition != -1):
                if (condition == 0):
                    return history_list[:-1], total_loss_list[:-1], total_val_loss_list[:-1], hidden_layer_data_list[-2]
                break
  
    return history_list, total_loss_list, total_val_loss_list, hidden_layer_data


def show_loss_plot(history_list, total_loss_list, total_val_loss_list):
    plt.plot([h.history['loss'] for h in history_list])
    plt.plot([h.history['val_loss'] for h in history_list])
    plt.plot(total_loss_list)
    plt.plot(total_val_loss_list)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test', 'Total_train', 'Total_test'], loc='upper left')
    plt.show()
    

def exit_condition(loss_list, min_delta):

    # len(lost_list)
    count = 0
    if (len(loss_list) < 2): return -1
    else:
        if (len(loss_list) <= 5): return -1
        elif (loss_list[-2] - loss_list[-1] < 0):
            return 0
        elif (loss_list[-2] - loss_list[-1] <= min_delta):
            return 1
    return -1
