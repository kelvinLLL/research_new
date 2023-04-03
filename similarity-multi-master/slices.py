import os
import sys
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

slice_idx = 0
total_index = []

def eachFile(path):
	fileList = os.listdir(path)
	for file in fileList:
		if os.path.isdir(path + "/" + file):
			eachFile(path + "/" + file)
		elif file[0] != '.':
			arr = file.split('.')
			if len(arr) > 1:
				if arr[len(arr)-1] == 'c' or arr[len(arr)-1] == 'cpp':
					f = open('sliceFile.txt', 'a', encoding='utf-8')
					f.writelines(path + "/" + file + "\n")
					f.close()
					os.system('python cut_slices.py sliceFile.txt')
					gather_slices_to_tmp()


def gather_slices_to_tmp():
	global slice_idx
	global total_index

	try:
		if os.path.exists('slice1.txt'):
		
			with open('index.txt', 'r', encoding='utf-8') as f:
				cur_index = f.readlines()
			if len(total_index) == 0:
				total_index.append(cur_index[0])
			for i in range(1, len(cur_index)):
				slice_info = cur_index[i].split()
				if len(slice_info) != 4:
					continue
				slice_idx += 1
				total_index.append('{:^5d}   {:^18s}   {:^14s}   {}\n'.format(slice_idx, slice_info[1], slice_info[2], slice_info[3]))
				os.system('mv slice' + slice_info[0] + '.txt tmp/slice' + str(slice_idx) + '.txt')
	finally:
		os.system('rm -f slice*.txt index.txt')


if __name__ == '__main__':
	if len(sys.argv) < 2:
		print("input format: python slices.py filepath")
	else:
		if os.path.exists('tmp'):
			os.system('rm -rf tmp')
		os.mkdir('tmp')
		filePath = sys.argv[1]
		if filePath[-1] == "/":
			filePath = filePath[0:-1]
		eachFile(filePath)
		with open('index.txt', 'w', encoding='utf-8') as f:
			f.writelines(total_index)
		os.system('mv tmp/slice*.txt .')
		os.system('rm -rf tmp')

		
