# -*- coding: utf-8 -*-
# + {}
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, \
    Concatenate, Flatten, SpatialDropout1D, \
    BatchNormalization, Conv1D, Maximum, ZeroPadding1D, \
    LSTM, Bidirectional
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.optimizers import Adam,RMSprop


import ahocorasick
from itertools import accumulate
import operator
import numpy as np
import copy as cp
from preprocessing import preprocess
prepro = preprocess()

from  tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_recall_fscore_support

f = open('SKut/variable/words_modified.txt')
dict_ = f.read().strip().split('\n')
A = ahocorasick.Automaton()
for idx, word in enumerate(dict_):
    A.add_word(word, len(word)) 
A.make_automaton()


# -

CHAR_TYPE = {
    u'กขฃคฆงจชซญฎฏฐฑฒณดตถทธนบปพฟภมยรลวศษสฬอ': 'c',
    u'ฅฉผฟฌหฝฮฤ': 'n',
    u'ัะาำิีืึุู': 'v',  # า ะ ำ ิ ี ึ ื ั ู ุ
    u'เแโใไ': 'w',
    u'่้๊๋็': 't', # วรรณยุกต์ ่ ้ ๊ ๋
    u'์ๆฯ.': 's', # ์  ๆ ฯ .
    u'0123456789๑๒๓๔๕๖๗๘๙': 'd',
    u'"': 'q',
    u"‘": 'q',
    u"’": 'q',
    u"'": 'q',
    u' ': 'p',
    u'<>`~๐;:-({)},./+*/-?!@#$%^&=][': 'p',
    u'abcdefghijklmnopqrstuvwxyz': 's_e',
    u'ABCDEFGHIJKLMNOPQRSTUVWXYZ': 'b_e'
}
CHAR_TYPE_FLATTEN = {}
for ks, v in CHAR_TYPE.items():
    for k in ks:
        CHAR_TYPE_FLATTEN[k] = v

CHAR_TYPES = [
    'b_e', 'c', 'd', 'n', 'o',
    'p', 'q', 's', 's_e', 't',
    'v', 'w'
]
CHAR_TYPES_MAP = {v: k for k, v in enumerate(CHAR_TYPES)}

# create map of dictionary to character
CHARS = [
    u'\n', u' ', u'!', u'"', u'#', u'$', u'%', u'&', "'", u'(', u')', u'*', u'+',
    u',', u'-', u'.', u'/', u'0', u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8',
    u'9', u':', u';', u'<', u'=', u'>', u'?', u'@', u'A', u'B', u'C', u'D', u'E',
    u'F', u'G', u'H', u'I', u'J', u'K', u'L', u'M', u'N', u'O', u'P', u'Q', u'R',
    u'S', u'T', u'U', u'V', u'W', u'X', u'Y', u'Z', u'[', u'\\', u']', u'^', u'_',
    u'a', u'b', u'c', u'd', u'e', u'f', u'g', u'h', u'i', u'j', u'k', u'l', u'm',
    u'n', u'o', u'other', u'p', u'q', u'r', u's', u't', u'u', u'v', u'w', u'x', u'y',
    u'z', u'}', u'~', u'ก', u'ข', u'ฃ', u'ค', u'ฅ', u'ฆ', u'ง', u'จ', u'ฉ', u'ช',
    u'ซ', u'ฌ', u'ญ', u'ฎ', u'ฏ', u'ฐ', u'ฑ', u'ฒ', u'ณ', u'ด', u'ต', u'ถ', u'ท',
    u'ธ', u'น', u'บ', u'ป', u'ผ', u'ฝ', u'พ', u'ฟ', u'ภ', u'ม', u'ย', u'ร', u'ฤ',
    u'ล', u'ว', u'ศ', u'ษ', u'ส', u'ห', u'ฬ', u'อ', u'ฮ', u'ฯ', u'ะ', u'ั', u'า',
    u'ำ', u'ิ', u'ี', u'ึ', u'ื', u'ุ', u'ู', u'ฺ', u'เ', u'แ', u'โ', u'ใ', u'ไ',
    u'ๅ', u'ๆ', u'็', u'่', u'้', u'๊', u'๋', u'์', u'ํ', u'๐', u'๑', u'๒', u'๓',
    u'๔', u'๕', u'๖', u'๗', u'๘', u'๙', u'‘', u'’', u'\ufeff'
]
CHARS_MAP = {v: k for k, v in enumerate(CHARS)}

# Character type
def get_ctype(c):
    for tag in tags:
        if c in tag[1]:
            return tag[0]
    return 'x'

def feature(x_text,entropy,prob):
    char_,type_ = prepro.create_feature_array(x_text,array=True) # text only
    
    dict_start_boundaries = set()
    dict_end_boundaries = set()
    for end_index, length in A.iter(x_text):
        start_index = end_index - length + 1
        dict_start_boundaries.add(start_index)
        dict_end_boundaries.add(end_index)
    
    addtional = []
    for i, char in enumerate(x_text):
        temp_ = []
        
        temp_.append(entropy[i])
        temp_.append(prob[i][1])
        if i in dict_start_boundaries:
            temp_.append(1)
        else:
            temp_.append(0)
        # If this character can be an end of word, according to our dictionary
        
        if i in dict_end_boundaries:
            temp_.append(1)
        else:
            temp_.append(0)
        
        addtional.append(temp_)
    
    addtional = pad_sequences(addtional, padding='post', maxlen=21, dtype = float)
    addtional_list = [np.array(x) for x in addtional]
    return char_,type_,np.array(addtional_list)

def conv_unit(inp, n_gram, no_word=200, window=2):
    out = Conv1D(no_word, window, strides=1, padding="valid", activation='relu')(inp)
    out = TimeDistributed(Dense(5, input_shape=(n_gram, no_word)))(out)
    out = ZeroPadding1D(padding=(0, window - 1))(out)
    return out


def get_convo_deepcut(no_word=200, n_gram=21, no_char=178):
    input1 = Input(shape=(n_gram,))
    input2 = Input(shape=(n_gram,))

    a = Embedding(no_char, 32, input_length=n_gram)(input1)
    a = SpatialDropout1D(0.15)(a)
    a = BatchNormalization()(a)

    a_concat = []
    for i in range(1,9):
        a_concat.append(conv_unit(a, n_gram, no_word, window=i))
    for i in range(9,12):
        a_concat.append(conv_unit(a, n_gram, no_word - 50, window=i))
    a_concat.append(conv_unit(a, n_gram, no_word - 100, window=12))
    a_sum = Maximum()(a_concat)

    b = Embedding(12, 12, input_length=n_gram)(input2)
    b = SpatialDropout1D(0.15)(b)

    x = Concatenate(axis=-1)([a, a_sum, b])
    #x = Concatenate(axis=-1)([a_sum, b])
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input1, input2], outputs=out)
    model.compile(optimizer=Adam(),
                  loss='binary_crossentropy', metrics=['acc'])
    return model

class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


def get_convo_nn2_lstm_attension(lstm_node,attention_node,optimizer,no_word=200, n_gram=21, no_char=178):
#     print('LSTM node:',lstm_node)
#     print('Attension:',attention_node)
    input1 = Input(shape=(n_gram,))
    input2 = Input(shape=(n_gram,))
    input3 = Input(shape=(n_gram,))
    
    a = Embedding(no_char, 32, input_length=n_gram)(input1)
    a = SpatialDropout1D(0.15)(a)
    a = BatchNormalization()(a)

    a_concat = []
    for i in range(1,9):
        a_concat.append(conv_unit(a, n_gram, no_word, window=i))
    for i in range(9,12):
        a_concat.append(conv_unit(a, n_gram, no_word - 50, window=i))
    a_concat.append(conv_unit(a, n_gram, no_word - 100, window=12))
    a_sum = Maximum()(a_concat)

    b = Embedding(12, 12, input_length=n_gram)(input2)
    b = SpatialDropout1D(0.15)(b)
    
    c = Embedding(12, 12, input_length=n_gram)(input3)
    c = SpatialDropout1D(0.15)(c)

    x = Concatenate(axis=-1)([a, a_sum, b,c])
    #x = Concatenate(axis=-1)([a_sum, b])
    x = BatchNormalization()(x)
    
    x,forward_h, forward_c, backward_h, backward_c = Bidirectional(LSTM(lstm_node,return_sequences=True,return_state=True))(x)
    
    state_h = Concatenate()([forward_h, backward_h])

    context_vector, attention_weights = Attention(attention_node)(x, state_h)
    
    #x = Flatten()(x)
    x = Dense(100, activation='relu')(context_vector)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input1, input2, input3], outputs=out)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', metrics=['acc'])
    return model


def eval_function(y_true,y_pred): # char level
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return fscore

def evaluate(train : list, test: list): # Word level
    train_acc = list(accumulate(map(len, train), func = operator.add))
    test_acc = list(accumulate(map(len, test), func = operator.add))
    train_set = set(zip([0,*train_acc], train_acc))
    test_set = set(zip([0,*test_acc], test_acc))
    correct = len(train_set & test_set)
    pre = correct/len(test)
    re = correct/len(test)
    f1 = (2*pre*re)/(pre+re)
    return f1

def cut(y_pred_boolean,x_data):
    x_ = cp.deepcopy(x_data)
    answer = []
    for idx,items in enumerate(y_pred_boolean):
        text = ""
        for index,item in enumerate(items):
            if(item == 1):
                text +='|'
            text +=x_[idx][index]
        answer.append(text)
    return answer 

def preprocess(x_data,y_true,y_ds_pred,y_dg_pred):
    dg_pred = cut([y_dg_pred],[x_data])
    true_pred = cut([y_true],[x_data])
    ds_pred = cut([y_ds_pred],[x_data])

    dg_list = dg_pred[0].split('|')
    true_list = true_pred[0].split('|')
    ds_list = ds_pred[0].split('|')

    f1_dg = evaluate(true_list,dg_list)
    f1_ds = evaluate(true_list,ds_list)
    return f1_dg,f1_ds
