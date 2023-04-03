import sys
import os
from utils import *

if len(sys.argv) != 3:
    print('Usage: python preprocess.py /input/file /output/directory')
    exit(1)

text = remove_annotation_singlefile(sys.argv[1])
with open('tmp.c', 'w') as f:
    f.writelines(text)
text = var_fun_transfer_singlefile('tmp.c')
if os.path.isdir(sys.argv[2]):
    os.system('rm -rf ' + sys.argv[2])
os.makedirs(sys.argv[2])
with open(sys.argv[2] + '/preprocessed.c', 'w') as f:
    f.writelines(text)
with open(sys.argv[2] + '/sliceFile.txt', 'w') as f:
    f.writelines('preprocessed.c\n')
os.system('cp cut_slices.py ' + sys.argv[2] + ' && cd ' + sys.argv[2] + ' && python cut_slices.py sliceFile.txt')
os.system('cd ' + sys.argv[2] + ' && rm -f cut_slices.py sliceFile.txt')

