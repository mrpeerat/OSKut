from preprocessing import preprocess
import copy as cp
import pickle
import numpy as np
prepro = preprocess()

# +
print('WS augmented Train.......')
test_list = ['wisesight-augmented']

# x,y_true = prepro.preprocess_x_y(test_list)
# y_true = [np.array(j) for sub in y_true for j in sub if len(j) > 1]
# x = [j for sub in x for j in sub if len(j) > 1]

# y_pred,y_entropy,y_prob = prepro.predict_(x,og='true') # DeepCut Baseline/BEST+WS/WS

pickle.dump(y_true, open('variable/y.true.ws-augment.train.vr', 'wb'))
pickle.dump(x, open(f'variable/x.ws-augment.train.vr', 'wb'))
pickle.dump(y_pred, open('variable/y.pred.ws-augment.train.vr', 'wb'))
pickle.dump(y_entropy, open('variable/y.entropy.ws-augment.train.vr', 'wb'))
pickle.dump(y_prob, open('variable/y.prob.ws-augment.train.vr', 'wb'))

# +
# print('BEST Train.......')
# #test_list = ['wisesight']
# #test_list = ['wisesight-testset']
# test_list = ['article','encyclopedia','news','novel']
# #test_list = ['testset']
# #test_list = ['tnhc_train']
# #test_list = ['tnhc_test']

# x,y_true = prepro.preprocess_x_y(test_list)
# y_true = [np.array(j) for sub in y_true for j in sub if len(j) > 1]
# x = [j for sub in x for j in sub if len(j) > 1]

# y_pred,y_entropy,y_prob = prepro.predict_(x,og='true') # DeepCut Baseline/BEST+WS/WS

# pickle.dump(y_pred, open('variable/y.pred.best.train.vr', 'wb'))
# pickle.dump(y_entropy, open('variable/y.entropy.best.train.vr', 'wb'))
# pickle.dump(y_prob, open('variable/y.prob.best.train.vr', 'wb'))

# +
# print('BEST Test.......')
# #test_list = ['wisesight']
# #test_list = ['wisesight-testset']
# #test_list = ['article','encyclopedia','news','novel']
# test_list = ['testset']
# #test_list = ['tnhc_train']
# #test_list = ['tnhc_test']

# x,y_true = prepro.preprocess_x_y(test_list)
# y_true = [np.array(j) for sub in y_true for j in sub if len(j) > 1]
# x = [j for sub in x for j in sub if len(j) > 1]

# y_pred,y_entropy,y_prob = prepro.predict_(x,og='true') # DeepCut Baseline/BEST+WS/WS

# pickle.dump(y_pred, open('variable/y.pred.best.test.vr', 'wb'))
# pickle.dump(y_entropy, open('variable/y.entropy.best.test.vr', 'wb'))
# pickle.dump(y_prob, open('variable/y.prob.best.test.vr', 'wb'))

# +
# print('TNHC Train.......')
# #test_list = ['wisesight']
# #test_list = ['wisesight-testset']
# #test_list = ['article','encyclopedia','news','novel']
# #test_list = ['testset']
# test_list = ['tnhc_train']
# #test_list = ['tnhc_test']

# x,y_true = prepro.preprocess_x_y(test_list)
# y_true = [np.array(j) for sub in y_true for j in sub if len(j) > 1]
# x = [j for sub in x for j in sub if len(j) > 1]

# y_pred,y_entropy,y_prob = prepro.predict_(x,og='true') # DeepCut Baseline/BEST+WS/WS

# pickle.dump(y_pred, open('variable/y.pred.tnhc.train.vr', 'wb'))
# pickle.dump(y_entropy, open('variable/y.entropy.tnhc.train.vr', 'wb'))
# pickle.dump(y_prob, open('variable/y.prob.tnhc.train.vr', 'wb'))

# +
# print('TNHC Test.......')
# #test_list = ['wisesight']
# #test_list = ['wisesight-testset']
# #test_list = ['article','encyclopedia','news','novel']
# #test_list = ['testset']
# #test_list = ['tnhc_train']
# test_list = ['tnhc_test']

# x,y_true = prepro.preprocess_x_y(test_list)
# y_true = [np.array(j) for sub in y_true for j in sub if len(j) > 1]
# x = [j for sub in x for j in sub if len(j) > 1]

# y_pred,y_entropy,y_prob = prepro.predict_(x,og='true') # DeepCut Baseline/BEST+WS/WS

# pickle.dump(y_pred, open('variable/y.pred.tnhc.test.vr', 'wb'))
# pickle.dump(y_entropy, open('variable/y.entropy.tnhc.test.vr', 'wb'))
# pickle.dump(y_prob, open('variable/y.prob.tnhc.test.vr', 'wb'))
# -

print('LST20 Train.......')
#test_list = ['wisesight']
#test_list = ['wisesight-testset']
#test_list = ['article','encyclopedia','news','novel']
#test_list = ['testset']
#test_list = ['tnhc_train']
#test_list = ['tnhc_test']
test_list = ['lst20']
x,y_true = prepro.preprocess_x_y(test_list)
y_true = [np.array(j) for sub in y_true for j in sub if len(j) > 1]
x = [j for sub in x for j in sub if len(j) > 1]
y_pred,y_entropy,y_prob = prepro.predict_(x,og='true') # DeepCut Baseline/BEST+WS/WS
pickle.dump(y_true, open('variable/y.true.lst20.train.vr', 'wb'))
pickle.dump(x, open(f'variable/x.lst20.train.vr', 'wb'))
pickle.dump(y_pred, open('variable/y.pred.lst20.train.vr', 'wb'))
pickle.dump(y_entropy, open('variable/y.entropy.lst20.train.vr', 'wb'))
pickle.dump(y_prob, open('variable/y.prob.lst20.train.vr', 'wb'))

print('LST20 Test.......')
#test_list = ['wisesight']
#test_list = ['wisesight-testset']
#test_list = ['article','encyclopedia','news','novel']
#test_list = ['testset']
#test_list = ['tnhc_train']
#test_list = ['tnhc_test']
test_list = ['lst20_test']
x,y_true = prepro.preprocess_x_y(test_list)
y_true = [np.array(j) for sub in y_true for j in sub if len(j) > 1]
x = [j for sub in x for j in sub if len(j) > 1]
y_pred,y_entropy,y_prob = prepro.predict_(x,og='true') # DeepCut Baseline/BEST+WS/WS
pickle.dump(y_true, open('variable/y.true.lst20.test.vr', 'wb'))
pickle.dump(x, open(f'variable/x.lst20.test.vr', 'wb'))
pickle.dump(y_pred, open('variable/y.pred.lst20.test.vr', 'wb'))
pickle.dump(y_entropy, open('variable/y.entropy.lst20.test.vr', 'wb'))
pickle.dump(y_prob, open('variable/y.prob.lst20.test.vr', 'wb'))
