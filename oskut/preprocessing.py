# -*- coding: utf-8 -*-
from pathlib import Path
from functools import reduce
import numpy as np
from scipy.stats import entropy
from sklearn.preprocessing import Normalizer
from tensorflow import keras
import oskut.deepcut as deepcut

class preprocess:
    repls = {'<NE>' : '','</NE>' : '','<AB>': '','</AB>': '','\n': '','\r': '','\r\n': '','\n\n':'','<p>':'','<s>':''}
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

    def read_file(self,path):
        words_all = []
        try:
            text = open(path,encoding='utf-8').readlines()
        except:
#             with open(path, 'rb') as f:
#                 text = f.readlines()
            raise Exception(f'Error file {path}')  
        for line in text:
            line = reduce(lambda a, kv: a.replace(*kv), self.repls.items(), line)
            words_all.append(line)
        return self.preprocess_attacut(words_all)

    def preprocess_attacut(self,sentence_lines):
        x = []
        y = []
        for sentence in sentence_lines:
            x.append(sentence.replace('|',''))
            sentence = '|' + sentence
            y_char = []
            for idx in range(1,len(sentence)):
                current_char = sentence[idx]
                before_char = sentence[idx-1]
                
                if current_char == '|':
                    continue
                
                target = 1 if before_char == '|' else 0  # y data
                y_char.append(target)
            y.append(y_char)
        
        return x,y
    
    def create_feature_array(self,text, n_pad=21,array=True): # Feature Exac from deepcut
        """
        Create feature array of character and surrounding characters
        """
        n = len(text)
        n_pad_2 = int((n_pad - 1)/2)
        text_pad = [' '] * n_pad_2  + [t for t in text] + [' '] * n_pad_2
        x_char, x_type = [], []
        for i in range(n_pad_2, n_pad_2 + n):
            char_list = text_pad[i + 1: i + n_pad_2 + 1] + \
                        list(reversed(text_pad[i - n_pad_2: i])) + \
                        [text_pad[i]]
            char_map = [self.CHARS_MAP.get(c, 80) for c in char_list]
            char_type = [self.CHAR_TYPES_MAP.get(self.CHAR_TYPE_FLATTEN.get(c, 'o'), 4)
                         for c in char_list]
            x_char.append(char_map)
            x_type.append(char_type)
        if array == True:
            x_char = np.array(x_char).astype(float)
            x_type = np.array(x_type).astype(float)
        return x_char, x_type

    def argmax_function(self,y):
        return [np.argmax(pred) for pred in y]

    def preprocess_x_y(self,test_list):
        context = []
        for folder in test_list:
            for filename in Path('corpus/'+folder).rglob('*.txt'):
                context.append(self.read_file(filename))
        x,y = list(zip(*context))
        return x,y

    def pred(self,data):
        preds = deepcut.tokenize(data)
        return preds

    def normalization(self,preds):
        return Normalizer().fit_transform(preds)

    def find_entropy(self,data_list):
        return [entropy(value,base=None) for value in data_list]

    def preprocessing_y_pred(self,y_pred):
        y_pred_ = []
        for sentence in y_pred:
            y_sentence = []
            for char_ in sentence:
                y_sentence.append([1-char_[0],char_[0]])
            y_pred_.append(y_sentence)
        return y_pred_
    
    def preprocessing_original(self,y_pred):
        y_pred_ = []
        for sentence in y_pred:
            y_sentence = []
            for char_ in sentence:
                try:
                    y_sentence.append([1-char_[1],char_[1]])
                except:
                    print(char_)
            y_pred_.append(y_sentence)
        return y_pred_

    def change_to_entropy(self,normalizae_data,random=False):
        if random:
            y_entropy = map(self.random_entropy,normalizae_data)  
        else:
            y_entropy = map(self.find_entropy,normalizae_data) #Random = False
        return list(y_entropy)

    def predict_(self,x,dl=False):
        x_char,x_type = self.create_feature_array(x)

        y_original_prob = list(map(self.pred,x))
        y_original_prepro = self.preprocessing_original(y_original_prob)
        y_norm_original = list(map(self.normalization,y_original_prepro))
        y_entropy_original = self.change_to_entropy(y_norm_original)
        y_original = list(map(self.argmax_function,y_original_prepro))

        if dl == True:
            pass
        else:
            return y_original,y_entropy_original,y_original_prob
