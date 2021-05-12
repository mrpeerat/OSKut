import warnings
warnings.filterwarnings('ignore')
import numpy as np
import copy as cp
import operator
from preprocessing import preprocess #Our class
prepro = preprocess()
import extract_features
from tensorflow.keras.optimizers import Adam,RMSprop



def load_model(engine='ws',mode='LSTM_Attension'):
    print('loading model.....')
    if engine != 'deepcut':
        optimizer = RMSprop(); opt_name = 'rmsprop'
        if engine == 'ws':
            lstm_node = 192
            attension_node = 96
        elif engine == 'tnhc':
            lstm_node = 192
            attension_node = 160
        elif engine == 'lst20':
            lstm_node = 224
            attension_node = 32
        elif 'tl-deepcut' in engine:
            mode = 'dg'
        elif 'ws-augment' in engine:
            lstm_node = 192
            attension_node = 32
        
        elif engine == 'deepcut_tnhc':
            mode = 'dg'
        else:
            raise Exception('Error engine')

        if mode == 'LSTM_Attension':
            lstm_node = lstm_node
            attension_node = attension_node
            model_load = extract_features.get_convo_nn2_lstm_attension(lstm_node,attension_node,optimizer)
        elif mode == 'LSTM':
            lstm_node = lstm_node
            attension_node = 0
            model_load = extract_features.get_convo_nn2_lstm(lstm_node,optimizer) 
        elif mode == 'Normal':
            lstm_node = 0
            attension_node = 0
            model_load = extract_features.get_convo_nn2(optimizer)
        elif mode == 'dg':
            model_load = extract_features.get_convo_deepcut()
        else:
            raise Exception('Error on Model')

        if mode == 'dg':
            try:
                engine_type = engine.split('-')[2:]
                engine_type = '-'.join(engine_type)
                print(engine_type)
                model_load.load_weights(f'SKut/weight/model_weight_{engine_type}.h5')
            except:
                print()
                raise Exception('Error Engine TL-XXXX-CORPUS_NAME')  
        else: 
            model_load.load_weights(f'SKut/model/ds_weights/{engine}_{opt_name}_{mode}_{lstm_node}_{attension_node}_weight.h5')
        global model; model = model_load
    else:
        pass
    global engine_mode; engine_mode = engine

def return_max_index(number_ranking,entropy_list):
    index_entropy = []
    func_entro_list = cp.deepcopy(entropy_list)
    ranking_ = int(len(entropy_list)*(number_ranking/100))
    #print(f'number we want is : {ranking_}')
    for i in range(ranking_):
        index, max_num = max(enumerate(func_entro_list), key=operator.itemgetter(1))
        func_entro_list[index] = -1.01
        index_entropy.append(index)
    return index_entropy

def scoring_function(entropy_before,x_function,y_dg_pred,y_entropy_function,y_prob_function,index):
    result = y_dg_pred[:]
    for i,items in enumerate(index):
        if items != []:
            char,char_type,addition = extract_features.feature(x_function[i],y_entropy_function[i],y_prob_function[i])
            if entropy_before != []:
                idx_cal = [*set(items)-set(entropy_before[i])]
                char_final = char[idx_cal]; char_type_final = char_type[idx_cal]; addition_final = addition[idx_cal]
            else:
                char_final = char[items]; char_type_final = char_type[items]; addition_final = addition[items]
                idx_cal = items
            try:
                y_pred_ds = model.predict([char_final,char_type_final,addition_final])
                ans = (y_pred_ds.ravel() > 0.5).astype(int)
                for idx,idx_item in enumerate(idx_cal):
                    result[i][idx_item] = ans[idx]
            except:
                pass
    return result

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

def SKut(sent,k=1):
    if type(sent) != list:
        sent = [sent]

    if k == 1:
        if engine_mode == 'lst20':
            k =  100
        elif engine_mode == 'tnhc':
            k =  100
        elif 'ws-augment' in engine_mode:
            k = 100
        else: #ws
            k = 33 #27
    
    if 'tl-deepcut' in engine_mode:
        y_pred=[]
        y_pred = [model.predict(prepro.create_feature_array(item)) for item in sent]
        y_pred_ = prepro.preprocessing_y_pred(y_pred)
        y_pred = list(map(prepro.argmax_function,y_pred_))
        x_answer = cut(y_pred,sent)
    else:
        y_original,y_entropy_original,y_prob_original = prepro.predict_(sent,og='true') #y_original = dg-model
        if engine_mode == 'deepcut':
            x_answer = cut(y_original,sent)
        else:
            entropy_index = [return_max_index(k,value) for value in y_entropy_original] # Find entropy index from DC Baseline
            entropy_before = []
            answer_ds_original = scoring_function(entropy_before,sent,y_original,y_entropy_original,y_prob_original,entropy_index) # Score function
            x_answer = cut(answer_ds_original,sent) 
    answer = x_answer[0].split('|')
    if answer[0] == '':
        return answer[1:]
    else:
        return answer


