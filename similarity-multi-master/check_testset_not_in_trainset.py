import os
import shutil

train_path = 'sard_10_classes/'
test_path = 'cwe121_test/'

for root, _, files in os.walk(test_path):
	for name in files:
		if os.path.isfile(train_path+name):
			print('duplicate: '+name)