#使用：python3 retrain_multi.py target_classification
#再训练：用户将新增的自定义训练数据放入目录add_sample/下，之后重新训练模型。
#样本文件名要求：文件名中包含cwe编号，并且包含good/bad标签。cwe编号按照预设的定义划入预设的10个分类，如果cwe编号不在这10类中，则划分为11类（其他类）。
#使用方法：新增用户数据后，对每一个目标类运行该脚本：python3 retrain_multi.py target_classification。其中target_classification是要训练的模型专门识别的类的编号（1-11）。
#可以将target_classification从1到11全部再训练一遍，也可以只再训练新增类对应的模型。

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, LSTM, Dense, Embedding, Concatenate, Dropout, Masking, Bidirectional
from keras.models import Model
from keras.layers.normalization import BatchNormalization
#from keras.utils import to_categorical
from keras.models import Sequential,Model
from keras.layers import Dense, Bidirectional, Input,Dropout
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.callbacks import Callback

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from gensim.models import word2vec
#from sklearn.model_selection import train_test_split
from utils import *

import sys
import numpy as np
import os
import random


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

#参数设定
num_lstm1 = 64
num_dense = 64
rate_drop_lstm = 0.1
rate_drop_dense = 0.1
MAX_SEQUENCE_LENGTH = 800
EMBEDDING_DIM = 30
split_number = 80000

path = "sard_10_classes/"
data,label = get_original_data(path)
path = "add_sample/"
data_added,label_added = get_original_data(path)
data = data + data_added
label = label + label_added

print("start tokenizer")
print("number of data",len(data))
#print(data[0][0:4])
print("exmaple of data:\n",data[0])
#print(type(data[0][0:4]))


class Attention(Layer):
	def __init__(self, step_dim=MAX_SEQUENCE_LENGTH,
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



#建立字典
if os.path.isfile('word_index.json'):
	word_index = read_data_from_json('word_index.json')
else:
	word_index = create_dictionary(data)
	write_data_to_json(word_index,'word_index.json')
	print("save dictionary word_index successfully")

#sequences
sequences = get_sequences(data,word_index)
#print('word_index is:',word_index)
print("example of data transferred to indexes:\n",sequences[0])
print('Found %s unique tokens.' % len(word_index))


pad_sequence,pad_label = pad_zero_to_sequences(sequences,label,MAX_SEQUENCE_LENGTH)

#把序列中的词编号转换成字符串，为word_vec做准备
for i in range(len(pad_sequence)):
	for j in range(len(pad_sequence[i])):
		pad_sequence[i][j] = str(pad_sequence[i][j])

#将pad_sequence,pad_label写入文件
write_data_to_json(pad_sequence,'pad_sequence.json')
write_data_to_json(pad_label,'pad_label.json')


print("start word2vec")
model = word2vec.Word2Vec(pad_sequence, min_count=3, size=EMBEDDING_DIM)
model.save("word.model")

#model = word2vec.Word2Vec.load("word.model")

print("creating enbedding index")
embeddings_index = {}
nb_words = len(word_index)+1
word_vectors = model.wv
for word, vocab_obj in model.wv.vocab.items():
	if int(vocab_obj.index) < nb_words:
		embeddings_index[word] = word_vectors[word]
#print(embeddings_index)
print("creating embedding matrix")
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
	if i >= nb_words:
		continue
	embedding_vector = embeddings_index.get(str(i))
	#print(embedding_vector)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector
print("for example,the first 5 words in dictionary will be transferred to this vector:\n",embedding_matrix[0:5])
print("example of sequence after padding",pad_sequence[0])


#根据标签区分正负样本
pos = []
neg = []
print("length of pad_sequence:",len(pad_sequence))
print("length of pad_label:",len(pad_label))
target_classification = int(sys.argv[1])
for i in range(len(pad_sequence)):
	if pad_label[i] == target_classification:
		pos.append(pad_sequence[i])
	else:
		neg.append(pad_sequence[i])
print("positive:",len(pos),"negative:",len(neg))
#p_sample = random.sample(pos, 10000)
#n_sample = random.sample(neg, 10000)
print("creating pairs")
pairs_1 = []
pairs_2 = []
pair_labels = []
for i in range(50000):
	x1 = random.choice(pos)
	y1 = random.choice(pos)
	pairs_1.append(x1)
	pairs_2.append(y1)
	pair_labels.append(1)

for i in range(50000):
	x1 = random.choice(neg)
	y1 = random.choice(pos)
	pairs_1.append(x1)
	pairs_2.append(y1)
	pair_labels.append(0)


seed = 4396
np.random.seed(seed)
np.random.shuffle(pairs_1)
np.random.seed(seed)
np.random.shuffle(pairs_2)
np.random.seed(seed)
np.random.shuffle(pair_labels)

#x_train, x_test, y_train, y_test = train_test_split(pairs, pair_labels, test_size=0.1, random_state=0)
x_train_1 = pairs_1[:split_number]
x_train_2 = pairs_2[:split_number]
x_test_1 = pairs_1[split_number:]
x_test_2 = pairs_2[split_number:]
y_train = pair_labels[:split_number]
y_test = pair_labels[split_number:]

x_train_1 = np.array(x_train_1)
x_train_2 = np.array(x_train_2)
y_train = np.array(y_train)
x_test_1 = np.array(x_test_1)
x_test_2 = np.array(x_test_2)
y_test = np.array(y_test)

print("number of training data:",len(x_train_1))
print("number of testing data:",len(x_test_1))


print("creating model")
def get_model():
#	masking_layer = Masking(mask_value=-1, input_shape=(200, 100))
	embedding_layer = Embedding(nb_words,
								EMBEDDING_DIM,
								weights=[embedding_matrix],
								input_length=MAX_SEQUENCE_LENGTH,
								trainable=False)
	first_lstm_layer1 = Bidirectional(LSTM(num_lstm1, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm,return_sequences=True))
#	second_lstm_layer1 = Bidirectional(LSTM(num_lstm2, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm))
	first_lstm_layer2 = Bidirectional(LSTM(num_lstm1, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm,return_sequences=True))
#	second_lstm_layer2 = Bidirectional(LSTM(num_lstm2, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm))
#	third_lstm_layer = Bidirectional(LSTM(num_lstm3, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm,return_sequences=True))
#	fourth_lstm_layer = Bidirectional(LSTM(num_lstm4, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm))


	input_1 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
	embedded_sequences_1 = embedding_layer(input_1)
	sequence_1_input = Masking(mask_value=0, input_shape=(MAX_SEQUENCE_LENGTH,EMBEDDING_DIM))(embedded_sequences_1)
	first_y1 = first_lstm_layer1(sequence_1_input)
	y1 =  Attention(MAX_SEQUENCE_LENGTH)(first_y1)
#	third_y1 = third_lstm_layer(second_y1)
#	y1 = fourth_lstm_layer(third_y1)
#	y1 = Dense(num_lstm4,activation = 'relu')(y1)

	input_2 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
	embedded_sequences_2 = embedding_layer(input_2)
	sequence_2_input = Masking(mask_value=0, input_shape=(MAX_SEQUENCE_LENGTH,EMBEDDING_DIM))(embedded_sequences_2)
	first_y2 = first_lstm_layer2(sequence_2_input)
	y2 = Attention(MAX_SEQUENCE_LENGTH)(first_y2)
#	third_y2 = third_lstm_layer(second_y2)
#	y2 = fourth_lstm_layer(third_y2)
#	y2 = Dense(num_lstm4,activation = 'relu')(y2)

	merged = Concatenate(axis = -1)([y1, y2])
	merged = Dropout(rate_drop_dense)(merged)
	merged = BatchNormalization()(merged)

	merged = Dense(num_dense, activation='relu')(merged)
	merged = Dropout(rate_drop_dense)(merged)
#	merged = BatchNormalization()(merged)
#	merged = Dense(16, activation='relu')(merged)
#	merged = Dropout(rate_drop_dense)(merged)
	merged = BatchNormalization()(merged)
#	merged = Dense(num_dense_2, activation='relu')(merged)
	preds = Dense(1, activation='sigmoid')(merged)

	model = Model(inputs=[input_1, input_2], outputs=preds)
	model.compile(loss='binary_crossentropy',
				  optimizer='adam',
				  metrics=['acc'])

	return model

print("shape of training data, format is (number of data,lenth of one sequence):",x_train_1.shape)


class Metrics(Callback):
	def on_train_begin(self, logs={}):
		self.val_f1s = []
		self.val_recalls = []
		self.val_precisions = []

	def on_epoch_end(self, epoch, logs={}):
		val_predict=(np.asarray(self.model.predict([x_test_1,x_test_2]))).round()
		val_targ = y_test
		_val_f1 = f1_score(val_targ, val_predict, average='binary')
		_val_recall = recall_score(val_targ, val_predict, average='binary')
		_val_precision = precision_score(val_targ, val_predict, average='binary')
		self.val_f1s.append(_val_f1)
		self.val_recalls.append(_val_recall)
		self.val_precisions.append(_val_precision)
		print("— _val_f1: %f "%_val_f1)
		print("— _val_recall: %f "%_val_recall)
		print("— _val_precision: %f "%_val_precision)
		return

metrics = Metrics()


model = get_model()
model.fit([x_train_1, x_train_2], y_train,
		  batch_size=32,
		  epochs=5,
		  validation_data=([x_test_1,x_test_2], y_test),
		  callbacks =[metrics])
print("saving model...")
model.save('models/'+str(sys.argv[1])+'_bilstm_retrained_model.h5')
print("model has been saved successfully")
