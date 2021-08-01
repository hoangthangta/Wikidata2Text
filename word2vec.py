#........................................................................................................
# Title: Wikidata claims (statements) to natural language (a part of Triple2Text/Ontology2Text task)
# Author: Ta, Hoang Thang
# Email: tahoangthang@gmail.com
# Lab: https://www.cic.ipn.mx
# Date: 12/2019
#........................................................................................................

from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import os

# word2vec using gensim ---------------------------------------------------------------------#
# min_count = 0, size = 150, sorted_vocab = 1, sg = 0 (0: CBOW - 1: Skip Gram), workers = 8
# p108:  iter (epoch) = 80, word_count = 1255, sentence_count = 231 
# p166:  iter (epoch) = 70, word_count = 1616, sentence_count = 334
# p26:   iter (epoch) = 50, word_count = 1474, sentence_count = 500
# p39:   iter (epoch) = 40, word_count = 1952, sentence_count = 594
# p937:  iter (epoch) = 30, word_count = 2093, sentence_count = 657
# p54:   iter (epoch) = 5,  word_count = 4795, sentence_count = 4621
# -------------------------------------------------------------------------------------------#

# call back function show accumulated loss and loss
class call_back(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.loss_to_be_subed = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        print('Loss after epoch {}: {} \t\t {}'.format(self.epoch, loss, loss_now))
        self.epoch += 1

# train and save word2vec to *.txt file
def save_word2vec(corpus, min_count, size, window, sorted_vocab, sg, workers, iters, file_name):
    model = Word2Vec(corpus, min_count = min_count, size = size, window = window, sorted_vocab = sorted_vocab, sg = sg,
                workers = workers, callbacks=[call_back()], compute_loss = True, iter = iters)
    path = os.path.dirname(os.path.abspath(__file__)) + "\\" + file_name
    model.wv.save_word2vec_format(path, binary=False)

# load word2vec from *.txt file
def load_word2vec(file_name):
    path = os.path.dirname(os.path.abspath(__file__)) + "\\" + file_name
    return KeyedVectors.load_word2vec_format(path, binary=False, unicode_errors='strict', encoding='utf8')

    
