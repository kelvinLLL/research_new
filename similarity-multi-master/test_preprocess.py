import sys
import os
from utils import *

if len(sys.argv) != 2:
    print('Usage: python test_preprocess.py input_path/')
    exit(1)

print("start preprocess ...")


input_path = sys.argv[1]
if input_path[-1] != '/':
	print('input path should be ended with /')
	exit(1)
output_path = input_path[:-1]+'_preprocessed/'
print("output path: ", output_path)

if not os.path.isdir(input_path):
	print("error: input path not exist.")
	exit(1)


if os.path.isdir(output_path):
    os.system('rm -rf ' + output_path)
os.makedirs(output_path)

remove_annotation_folder(input_path,"test_preprocess_tmp1/")

if os.path.isdir("test_preprocess_tmp2/"):
	os.system('rm -rf ' + "test_preprocess_tmp2/")
os.makedirs("test_preprocess_tmp2/")

os.system('cp cut_slices.py slices.py ' + 'test_preprocess_tmp2/' + ' && cd ' + 'test_preprocess_tmp2/' + ' && python slices.py ../test_preprocess_tmp1/')
os.system('cd ' + "test_preprocess_tmp2/" + ' && rm -f cut_slices.py slices.py index.txt')


var_fun_transfer_folder("test_preprocess_tmp2/",output_path)

os.system("rm -rf test_preprocess_tmp1/ test_preprocess_tmp2/")
print("preprocess finished.")

