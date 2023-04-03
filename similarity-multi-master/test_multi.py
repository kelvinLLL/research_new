#usage:python3 test_multi.py testpath
#对testpath目录下的若干个样本进行测试，将测试结果输出到csv中。

#from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, LSTM, Dense, Embedding, Concatenate, Dropout, Masking, Bidirectional, Conv1D, MaxPooling1D, Flatten
from keras.models import Model,load_model
from keras.layers.normalization import BatchNormalization
from gensim.models import word2vec
from keras.utils import to_categorical
from keras.models import Sequential,Model

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from utils import *

import numpy as np
import os
import random
import sys
import json
import csv
import heapq

MAX_SEQUENCE_LENGTH = 800
EMBEDDING_DIM = 30

path = "sard_10_classes/"
testpath = sys.argv[1]
modelpath = "models/"

#不输出warning信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#禁用gpu，强制使用cpu（因为nvidia驱动版本太老，尚未更新，临时设置此项）
os.environ["CUDA_VISIBLE_DEVICES"] = ""

class Attention(Layer):
	def __init__(self,step_dim=MAX_SEQUENCE_LENGTH,
				 W_regularizer=None, b_regularizer=None,
				 W_constraint=None, b_constraint=None,
				 bias=True, **kwargs):
		self.supports_masking = True
		self.init = initializers.get('glorot_uniform')

		self.W_regularizer = regularizers.get(W_regularizer)
		self.b_regularizer = regularizers.get(b_regularizer)

		self.W_constraint = constraints.get(W_constraint)
		self.b_constraint = constraints.get(b_constraint)

		self.bias = bias
		self.step_dim = step_dim
		self.features_dim = 0
		super(Attention, self).__init__(**kwargs)

	def build(self, input_shape):
		assert len(input_shape) == 3

		self.W = self.add_weight(shape=(input_shape[-1],),
								 initializer=self.init,
								 name='{}_W'.format(self.name),
								 regularizer=self.W_regularizer,
								 constraint=self.W_constraint)
		self.features_dim = input_shape[-1]

		if self.bias:
			self.b = self.add_weight(shape=(input_shape[1],),
									 initializer='zero',
									 name='{}_b'.format(self.name),
									 regularizer=self.b_regularizer,
									 constraint=self.b_constraint)
		else:
			self.b = None

		self.built = True

	def compute_mask(self, input, input_mask=None):
		return None

	def call(self, x, mask=None):
		features_dim = self.features_dim
		step_dim = self.step_dim

		eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
						K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

		if self.bias:
			eij += self.b

		eij = K.tanh(eij)

		a = K.exp(eij)

		if mask is not None:
			a *= K.cast(mask, K.floatx())

		a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

		a = K.expand_dims(a)
		weighted_input = x * a
		return K.sum(weighted_input, axis=1)

	def compute_output_shape(self, input_shape):
		return input_shape[0],  self.features_dim

if not os.path.isfile('word_index.json'):
	print('word_index.json does not exist.')
	exit()
else:
	word_index = read_data_from_json('word_index.json')
nb_words = len(word_index)+1


if not os.path.isfile('test_compare_sample.json'):
	if not os.path.isfile('pad_sequence.json'):
		print('pad_sequence.json does not exist.')
		exit()
	else:
		pad_sequence = read_data_from_json('pad_sequence.json')

	if not os.path.isfile('pad_label.json'):
		print('pad_label.json does not exist.')
		exit()
	else:
		pad_label = read_data_from_json('pad_label.json')

	#now we have pad_sequence and pad_label

	test_compare_sample = create_test_compare_sample(pad_sequence,pad_label)
	write_data_to_json(test_compare_sample,'test_compare_sample.json')
else:
	test_compare_sample = read_data_from_json('test_compare_sample.json')


target_pad_sequences = []
#target_pad_labels = []
test_file_names = []

#对testpath中的切片做变量名替换，并将文件夹名字加上后缀_transferred。
if testpath[-1] != '/':
	print("error:testpath format should be xxxx/")
	exit()
elif os.path.isdir(testpath[:-1]+"_transferred/"):
	print("已经做过变量名替换。")
	testpath = testpath[:-1]+"_transferred/"
else:
	var_fun_transfer_folder(testpath,testpath[:-1]+"_transferred/")
	testpath = testpath[:-1]+"_transferred/"


for root, _, files in os.walk(testpath):
	for name in files:
		target_pad_sequences.append(test_target_preprocess(os.path.join(root, name),word_index,MAX_SEQUENCE_LENGTH))
#		target_pad_labels.append(get_label(name))
		test_file_names.append(name)

print('Total number of target_pad_sequences :',len(target_pad_sequences))


#total_score是一个二维数组，外层表示对应的类别，内层表示每个样本对应该类别的可能性评分（0-1之间）。
total_score = []
for kind in range(1,11):
	print('Loading model:',str(kind)+'_bilstm_model.h5')
	model = load_model(modelpath+str(kind)+'_bilstm_model.h5',custom_objects={"Attention": Attention})
	pairs_1 = []
	pairs_2 = []
	for i in range(len(target_pad_sequences)):
		for j in range((kind-1)*5,kind*5):
			pairs_1.append(target_pad_sequences[i])
			pairs_2.append(test_compare_sample[j])
	print('Start predicting with model:',str(kind)+'_bilstm_model.h5')
	score_list_1 = model.predict([pairs_1,pairs_2], batch_size=100, verbose=0)
	score_list_1 = score_list_1.reshape(-1)
	print('Predicting with model:',str(kind)+'_bilstm_model.h5',' finished.')
	score_list_2 = []
	for i in range(len(target_pad_sequences)):
		#保留五位小数
		score_list_2.append(round(sum(score_list_1[i*5:(i+1)*5])/5.0,5))
	print('kind ',kind,'results finished. Next model.')
	total_score.append(score_list_2)


number_of_samples = len(target_pad_sequences)
'''
TP = 0
FP = 0
TN = 0
FN = 0
'''
f = open('results.csv','w',encoding='utf-8',newline='')
csv_writer = csv.writer(f)
csv_writer.writerow(["filename","result_top1","result_top2","result_top3","score1","score2","score3","score4","score5","score6","score7","score8","score9","score10"])

for i in range(len(target_pad_sequences)):
	one_sample_score = []
	for j in range(len(total_score)):
		one_sample_score.append(total_score[j][i])
	one_sample_top3_score = heapq.nlargest(3, one_sample_score)
	one_sample_top3_index = list(map(one_sample_score.index, heapq.nlargest(3, one_sample_score)))
	if one_sample_top3_score[0] > 0.5:
		one_sample_result = [one_sample_top3_index[0]+1,one_sample_top3_index[1]+1,one_sample_top3_index[2]+1]
	else:
		one_sample_result = [0,0,0]
	csv_writer.writerow([test_file_names[i],one_sample_result[0],one_sample_result[1],one_sample_result[2]]+one_sample_score)
'''
	if one_sample_result == [0,0,0] and target_pad_labels[i] == 0:
		TN += 1
	elif one_sample_result != [0,0,0] and target_pad_labels[i] in one_sample_result:
		TP += 1
	elif one_sample_result == [0,0,0] and target_pad_labels[i] != 0:
		FN += 1
	else:
		FP += 1
'''
print('Write csv finished.')
f.close()

'''
#准确率
accuracy = (TP+TN)/(number_of_samples)
#检出率
recall = TP/(TP+FN)
#误报率
false_positive_rate = FP/(FP+TP)
#打印结果
print('===================测试结果===================')
print('测试结果已写入文件：results.csv')
print('TP=',TP,' TN=',TN,' FP=',FP,' FN=',FN)
print('检出率：',recall)
print('误报率：',false_positive_rate)
print('准确率：',accuracy)
print('==============================================')
'''