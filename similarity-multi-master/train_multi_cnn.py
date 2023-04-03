#usage:python3 train_multi_cnn.py

#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, LSTM, Dense, Embedding, Concatenate, Dropout, Masking, Bidirectional, Conv1D, MaxPooling1D, Flatten
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
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
	

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#参数设定


MAX_SEQUENCE_LENGTH = 800
EMBEDDING_DIM = 30



path = "sard_10_classes/"
data,label = get_original_data(path)


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


#将训练集中的标签为0的样本进行采样，保留7000个。
pos = []
pos_label = []
neg = []
for i in range(len(pad_sequence)):
	if pad_label[i] == 0:
		neg.append(pad_sequence[i])
	else:
		pos.append(pad_sequence[i])
		pos_label.append(pad_label[i])
neg = random.sample(neg, 7000)
pos.extend(neg)
pos_label.extend([0]*7000)

pad_sequence = pos
pad_label = pos_label

seed = 4396
np.random.seed(seed)
np.random.shuffle(pad_sequence)
np.random.seed(seed)
np.random.shuffle(pad_label)
print('已经对标签为0的样本进行欠采样，pad_sequence长度：',len(pad_sequence),',pad_label长度：',len(pad_label))


#划分训练集和测试集

split_number = int(len(pad_sequence)*0.8)
#x_train, x_test, y_train, y_test = train_test_split(pairs, pair_labels, test_size=0.1, random_state=0)

x_train = pad_sequence[:split_number]
x_test = pad_sequence[split_number:]
y_train = pad_label[:split_number]
y_test = pad_label[split_number:]


x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

print("number of training data:",len(x_train))
print("number of testing data:",len(x_test))


print("creating model")

def get_model():

	embedding_layer = Embedding(nb_words,
								EMBEDDING_DIM,
								weights=[embedding_matrix],
								input_length=MAX_SEQUENCE_LENGTH,
								trainable=False)

	main_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
	embed = embedding_layer(main_input)
	

	# 词窗大小分别为3,4,5
	cnn1 = Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
	cnn1 = MaxPooling1D(pool_size=EMBEDDING_DIM)(cnn1)
	cnn2 = Conv1D(256, 4, padding='same', strides=1, activation='relu')(embed)
	cnn2 = MaxPooling1D(pool_size=EMBEDDING_DIM)(cnn2)
	cnn3 = Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
	cnn3 = MaxPooling1D(pool_size=EMBEDDING_DIM)(cnn3)
	# 合并三个模型的输出向量
	cnn = Concatenate(axis=-1)([cnn1, cnn2, cnn3])
	flat = Flatten()(cnn)
	drop = Dropout(0.2)(flat)
	main_output = Dense(11, activation='softmax')(drop)


	model = Model(inputs=main_input, outputs=main_output)
	model.compile(loss='categorical_crossentropy',
				  optimizer='adam',
				  metrics=['accuracy'])

	return model

print("shape of training data, format is (number of data,lenth of one sequence):",x_train.shape)


class Metrics(Callback):
	def on_train_begin(self, logs={}):
		self.val_f1s = []
		self.val_recalls = []
		self.val_precisions = []

	def on_epoch_end(self, epoch, logs={}):
		val_predict = np.argmax(np.asarray(self.model.predict(self.validation_data[0])), axis=1)
		val_targ = np.argmax(self.validation_data[1], axis=1)
		_val_f1 = f1_score(val_targ, val_predict, average='macro')
		#_val_recall = recall_score(val_targ, val_predict, average='binary')
		#_val_precision = precision_score(val_targ, val_predict, average='binary')
		self.val_f1s.append(_val_f1)
		#self.val_recalls.append(_val_recall)
		#self.val_precisions.append(_val_precision)
		print("— _val_f1: %f "%_val_f1)
		#print("— _val_recall: %f "%_val_recall)
		#print("— _val_precision: %f "%_val_precision)
		return

metrics = Metrics()



model = get_model()

one_hot_labels_train = to_categorical(y_train, num_classes=11)  # 将标签转换为one-hot编码
one_hot_labels_test = to_categorical(y_test, num_classes=11)  # 将标签转换为one-hot编码
model.fit(x_train, one_hot_labels_train,
		  batch_size=32,
		  epochs=5,
		  class_weight='auto',
		  validation_data=(x_test, one_hot_labels_test),
		  callbacks =[metrics])
print("saving model...")
model.save('cnn_model.h5')
print("model has been saved successfully")