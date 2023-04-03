#usage:python3 predict_ll.py filename

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, LSTM, Dense, Embedding, Concatenate, Dropout, Masking, Bidirectional
from keras.models import Model,load_model
from keras.layers.normalization import BatchNormalization
#from keras.utils import to_categorical
#from gensim.models import word2vec
#from sklearn.model_selection import train_test_split
from keras.models import Sequential,Model
from keras.layers import Dense, Bidirectional, Input,Dropout
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

import numpy as np
import os
import random
import sys
from utils import *

MAX_SEQUENCE_LENGTH = 800
EMBEDDING_DIM = 30
split_number = 80000

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



target_pad_sequence = test_target_preprocess(sys.argv[1],word_index,MAX_SEQUENCE_LENGTH)

#print("padding OK")


for j in range(len(target_pad_sequence)):
	target_pad_sequence[j] = str(target_pad_sequence[j])


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





def predict_one_kind_of_vul(kind):
	pairs_1 = []
	pairs_2 = []

	for i in range((kind-1)*5,kind*5):
		x1 = test_compare_sample[i]
		pairs_1.append(target_pad_sequence)
		pairs_2.append(x1)
	pairs_1 = np.array(pairs_1)
	pairs_2 = np.array(pairs_2)
	model = load_model(str(kind)+'_bilstm_model.h5',custom_objects={"Attention": Attention})
	score_list = model.predict([pairs_1,pairs_2], batch_size=5, verbose=0)
	score_list = score_list.reshape(-1)
	print(score_list)
	score = sum(score_list)/5.0
	print("kind ",kind," score is:",score)
	return score

"""
CWE分类：根据已有数据集，分为以下类别：(含训练准确率)
	1. buffer_read_or_write_out_of_bounds	'CWE121', 'CWE122', 'CWE123', 'CWE124', 'CWE126', 'CWE127'  acc:95.94% recall:99.93% precision:93.51% f1:0.9662
	2. integer_overflow	'CWE190'		acc:98.87% recall:99.69% precision:98.24% f1:0.9911
	3. integer_underflow	'CWE191'	acc:98.83% recall:99.85% precision:97.97% f1:0.9895
	4. mismatched_memory_management_routines	'CWE762'		acc:99.69% recall:99.83% precision:99.68% f1:0.9984
	5. free_memory_not_on_heap	'CWE590'		acc:99.79% recall:99.89% precision:99.70% f1:0.9985
	6. resource_and_memory_exhaustion	'CWE400', 'CWE401', 'CWE404'		acc:98.81% recall:99.68% precision:98.43% f1:0.9905
	7. Use of externally-controlled format string    'CWE134'	acc:99.86% recall:99.97% precision:99.78% f1:0.9987
	8. problems_with_signed_to_unsigned_transformation_and_numeric_lenth	'CWE194', 'CWE195', 'CWE196', 'CWE197'		acc:98.78% recall:99.88% precision:97.98% f1:0.9897
	9. use_of_uninitialized_variable	'CWE457'		acc:99.59% recall:99.89% precision:99.37% f1:0.9963
	10. problems_during_free_operation	'CWE415', 'CWE416'		acc:99.58% recall:99.95% precision:99.61% f1:0.9980
	11. others
"""

results = {}
for i in range(1,11):
	results[i] = predict_one_kind_of_vul(i)
print("====================score overview========================")
print("1. buffer_read_or_write_out_of_bounds\t\t",results[1])
print("2. integer_overflow\t\t",results[2])
print("3. integer_underflow\t\t",results[3])
print("4. mismatched_memory_management_routines\t\t",results[4])
print("5. free_memory_not_on_heap\t\t",results[5])
print("6. resource_and_memory_exhaustion\t\t",results[6])
print("7. Use of externally-controlled format string\t\t",results[7])
print("8. problems_with_signed_to_unsigned_transformation_and_numeric_length\t\t",results[8])
print("9. use_of_uninitialized_variable\t\t",results[9])
print("10. problems_during_free_operation\t\t",results[10])
print("==========================================================")
#if the max score is less than 0.5, the target is not vulnerable
if results[max(results, key=results.get)] < 0.5:
	print("not vulnerable")
else:
	print("vulnerability classification:",max(results, key=results.get))
#print results
