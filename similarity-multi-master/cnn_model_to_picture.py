#usage:python3 cnn_model_to_picture.py model_name
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.utils import plot_model
from keras.models import load_model
from keras import backend as K
import os
import sys

MAX_SEQUENCE_LENGTH = 800

#禁用gpu，强制使用cpu（因为nvidia驱动版本太老，尚未更新，临时设置此项）
os.environ["CUDA_VISIBLE_DEVICES"] = ""
#不输出warning信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

'''
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
'''


model_name = "models/"+sys.argv[1]
if os.path.isfile(model_name):
	model = load_model(model_name)
	plot_model(model, to_file = 'model_pictures/'+'cnn_model'+'.png')
else:
	print('file: '+model_name+' does not exist.')
	exit()