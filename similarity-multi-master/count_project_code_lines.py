#usage: python count_project_code_lines.py path/

import sys
import os
import re

if (len(sys.argv) != 2):
	print('usage: python count_project_code_lines.py path/')
	exit()

input_path = sys.argv[1]
count = 0

if not os.path.isdir(input_path):
	print('input path not exist.')
	exit()

for root, _,files in os.walk(input_path):
	for name in files:
		if re.findall(r'\.cpp$|\.c$|\.h$',name):
			print('filename:',name)
			f = open(os.path.join(root,name),encoding="utf-8",errors='ignore')
			count += len(f.readlines())


print("total number of code lines: ", count)