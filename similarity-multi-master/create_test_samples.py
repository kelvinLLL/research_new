#usage: python3 create_test_samples.py
#采用随机抽取的方式，构造由1000个样本组成的测试集。
import random
import os
import shutil


def sample_rough_classification(name):
	list_1 = ['CWE121', 'CWE122', 'CWE123', 'CWE124', 'CWE126', 'CWE127']
	for cwe in list_1:
		if cwe in name:
			return 1
	if 'CWE190' in name:
		return 2
	elif 'CWE191' in name:
		return 3
	elif 'CWE762' in name:
		return 4 
	elif 'CWE590' in name:
		return 5
	elif ('CWE400' in name) or ('CWE401' in name) or ('CWE404' in name):
		return 6
	elif 'CWE789' in name:
		return 7
	elif ('CWE194' in name) or ('CWE195' in name) or ('CWE196' in name) or ('CWE197' in name):
		return 8
	elif 'CWE457' in name:
		return 9
	elif ('CWE415' in name) or ('CWE416' in name):
		return 10
	else:
		return 11



def random_copyfile(srcPath,dstPath,numfiles):
	name_list=list(os.path.join(srcPath,name) for name in os.listdir(srcPath))
	name_list_2 = []
	for i in range(len(name_list)):
		if sample_rough_classification(name_list[i]) != 11 and os.path.getsize(name_list[i]) < 4000:
			name_list_2.append(name_list[i])
	random_name_list=list(random.sample(name_list_2,numfiles))
	if not os.path.exists(dstPath):
		os.mkdir(dstPath)
	for oldname in random_name_list:
		shutil.copyfile(oldname,oldname.replace(srcPath, dstPath))

srcPath='sard_cwe121/'		 
dstPath = 'sard_cwe121_test/'
random_copyfile(srcPath,dstPath,2000)

print('finished.')
