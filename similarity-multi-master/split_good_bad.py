#usage: python split_good_bad.py 
from utils import *
import os
from time import *

split_good_bad_folder('cwe121_test/','cwe121_test_splitted/')
sleep(5)
os.system('rm -rf cwe121_test/')
remove_annotation_folder('cwe121_test_splitted/','cwe121_test/')
sleep(5)
os.system('rm -rf cwe121_test_splitted/')