#usage:python3 predict_multi_cnn.py filename

from keras.layers import Input, LSTM, Dense, Embedding, Concatenate, Dropout, Masking, Bidirectional, Conv1D, MaxPooling1D, Flatten
from keras.models import Model,load_model
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras.models import Sequential
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.callbacks import Callback

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
#from gensim.models import word2vec
#from sklearn.model_selection import train_test_split
from utils import *

import sys
import numpy as np
import os
import random

MAX_SEQUENCE_LENGTH = 800
EMBEDDING_DIM = 30

"""
CWE分类：根据已有数据集，分为以下类别：
	1. buffer_read_or_write_out_of_bounds	'CWE121', 'CWE122', 'CWE123', 'CWE124', 'CWE126', 'CWE127'
	2. integer_overflow	'CWE190'
	3. integer_underflow	'CWE191'
	4. mismatched_memory_management_routines	'CWE762'
	5. free_memory_not_on_heap	'CWE590'
	6. resource_and_memory_exhaustion	'CWE400', 'CWE401', 'CWE404'
	7. Use of externally-controlled format string	'CWE134'
	8. problems_with_signed_to_unsigned_transformation_and_numeric_lenth	'CWE194', 'CWE195', 'CWE196', 'CWE197'
	9. use_of_uninitialized_variable	'CWE457'
	10. problems_during_free_operation	'CWE415', 'CWE416'
	11. others
"""

#不输出warning信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#禁用gpu，强制使用cpu（因为nvidia驱动版本太老，尚未更新，临时设置此项）
os.environ["CUDA_VISIBLE_DEVICES"] = ""

if not os.path.isfile('word_index.json'):
	print('word_index.json does not exist.')
	exit()
else:
	word_index = read_data_from_json('word_index.json')
nb_words = len(word_index)+1

target_pad_sequence = test_target_preprocess(sys.argv[1],word_index,MAX_SEQUENCE_LENGTH)

#for j in range(len(target_pad_sequence)):
#	target_pad_sequence[j] = str(target_pad_sequence[j])


model = load_model('cnn_model.h5')
target_pad_sequence = np.array([target_pad_sequence])
score_list = model.predict(target_pad_sequence, batch_size=1, verbose=0)
score_list = score_list.reshape(-1)

#print(score_list)

print("====================score overview========================")
print("1. buffer_read_or_write_out_of_bounds\t\t",score_list[1])
print("2. integer_overflow\t\t",score_list[2])
print("3. integer_underflow\t\t",score_list[3])
print("4. mismatched_memory_management_routines\t\t",score_list[4])
print("5. free_memory_not_on_heap\t\t",score_list[5])
print("6. resource_and_memory_exhaustion\t\t",score_list[6])
print("7. Use of externally-controlled format string\t\t",score_list[7])
print("8. problems_with_signed_to_unsigned_transformation_and_numeric_length\t\t",score_list[8])
print("9. use_of_uninitialized_variable\t\t",score_list[9])
print("10. problems_during_free_operation\t\t",score_list[10])
print("==========================================================")
#if score_list[0] is more than 0.5, the target is not vulnerable
if score_list[0] > 0.5:
	print("not vulnerable")
else:
	print("vulnerability classification:",np.argmax(score_list))
#print results

