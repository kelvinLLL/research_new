#usage: python generate_model_pictures.py
import os
import sys

os.system("python bilstm_model_to_picture.py lstm_test.h5")
print("finished.")
os.system("python cnn_model_to_picture.py cnn_test.h5")
print("finished.")