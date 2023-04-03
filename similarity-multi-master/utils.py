import sys
import os
import re
import json
import random

#sard中的代码样本分为good版本和bad版本，可以根据编译选项将其编译成good版或bad版，分别对应无漏洞和有漏洞。
#为了将sard中的代码样本用于模型训练，需要将样本切分成good的部分和bad的部分，切分后变成两个样本。
#该函数对单个代码样本进行切分。
#输入：input_filename=文件名。
#输出：返回两个list，一个是good一个是bad。
def split_good_bad_singlefile(input_filename):
	inputTemp = open(input_filename,"r",encoding="utf-8")
	#outputGoodTemp=open(input_filename.split(".")[0]+'_good.txt',"w",encoding="utf-8")
	#outputBadTemp=open(input_filename.split(".")[0]+'_bad.txt',"w",encoding="utf-8")
	inGood = 0
	inBad = 0
	lineItems = inputTemp.readlines()
	outputGoodTemp = []
	outputBadTemp = []
	for lineItem in lineItems:
		if ("OMITGOOD" in lineItem) and ("#ifndef" in lineItem) and (lineItem[0]=='#'):
			inGood = 1
		elif ("OMITBAD" in lineItem) and ("#ifndef" in lineItem) and (lineItem[0]=='#'):
			inBad = 1
		elif ("OMITGOOD" in lineItem) and ("#endif" in lineItem) and (lineItem[0]=='#'):
			inGood = 0
		elif ("OMITBAD" in lineItem) and ("#endif" in lineItem) and (lineItem[0]=='#'):
			inBad = 0
		else:
			if not inBad:
				outputGoodTemp.append(lineItem)
			if not inGood:
				outputBadTemp.append(lineItem)
	inputTemp.close()
	#outputGoodTemp.close()
	#outputBadTemp.close()
	return outputGoodTemp,outputBadTemp


#该函数对整个文件夹中的漏洞样本进行批量化处理，将每个样本切分成good和bad两个，并且存放到目标文件夹下。
#输入：input_path为输入样本的路径（文件夹），output_path为切分后的样本的输出路径（文件夹）
def split_good_bad_folder(input_path,output_path):
	if not os.path.isdir(input_path):
		print('input_path does not exist.')
		return 1
	if not os.path.isdir(output_path):
		os.makedirs(output_path)
		print('output_path does not exist, a new folder has been created.')
	for root, _, files in os.walk(input_path):
		os.chdir(input_path)
		for name in files:
			outputGoodStr,outputBadStr = split_good_bad_singlefile(name)
			outputGoodTemp=open('../'+output_path+'/'+name.split('.')[0]+'_good.txt',"w",encoding="utf-8")
			outputBadTemp=open('../'+output_path+'/'+name.split('.')[0]+'_bad.txt',"w",encoding="utf-8")
			outputGoodTemp.writelines(outputGoodStr)
			outputBadTemp.writelines(outputBadStr)
			outputGoodTemp.close()
			outputBadTemp.close()
		os.chdir('../')
	print('split_good_bad_folder finished.')
	return 0


#该函数对单个漏洞样本进行去注释处理。输入一个文本形式的漏洞样本，输出去注释版本的样本。
#输入：input_filename
#输出：返回去注释后的样本outputTemp，以list形式。接下来可用writelines写入文件。
def remove_annotation_singlefile(input_filename):
	inputTemp = open(input_filename,"r",encoding="utf-8")
	lineItems = inputTemp.readlines()
	outputTemp = []
	for lineItem in lineItems:
		if ("/*" in lineItem) and ("*/" not in lineItem):
			anno1 = 1
		elif ("*/" in lineItem) and ("/*" not in lineItem):
			anno1 = 0
		elif ("/*" in lineItem) and ("*/" in lineItem):
			lineItem_list = lineItem.split("/*")
			lineItem =lineItem_list[0]+'\n'
			outputTemp.append(lineItem)
		elif "//" in lineItem:
			lineItem_list = lineItem.split("//")
			lineItem =lineItem_list[0]+'\n'
			outputTemp.append(lineItem)
		else:
			if not anno1:
				outputTemp.append(lineItem)
	inputTemp.close()
	return outputTemp


#该函数对整个文件夹中的样本进行批量化去注释处理。
#输入：input_path为输入样本的路径（文件夹），output_path为去注释后的样本的输出路径（文件夹）。
def remove_annotation_folder(input_path,output_path):
	if not os.path.isdir(input_path):
		print('input_path does not exist.')
		return 1
	if not os.path.isdir(output_path):
		os.makedirs(output_path)
		print('output_path does not exist, a new folder has been created.')
	for root, _, files in os.walk(input_path):
		os.chdir(input_path)
		for name in files:
			outputStr = remove_annotation_singlefile(name)
			outputTemp=open('../'+output_path+'/'+name,"w",encoding="utf-8")
			outputTemp.writelines(outputStr)
			outputTemp.close()
		os.chdir('../')
	print('remove_annotation_folder finished.')
	return 0


#该函数判定一个字符串是否是一个identifier。一个identifier由字母，数字，和下划线组成，并且第一个字符不能是数字。若该字符串是identifier，则返回1，否则返回0。
#输入：字符串s
#输出：是identifier则返回1，否则返回0。
def is_identifier(s):
	if s[0] == '_' or s[0].isalpha():
		for i in s:
			if i == '_' or i.isalpha() or i.isdigit():
				pass
			else:
				return 0
		return 1
	else:
		return 0


#该函数对单个文件进行变量名替换处理。将所有程序员定义的变量名，函数名，及类名替换成统一的形式，如VAR1,FUN1等等。
#输入：input_filename
#输出：返回一个字符串write_sequence，接下来可以写入到文件里，也可以输出到控制台
def var_fun_transfer_singlefile(input_filename):
	if not os.path.isfile('wordlist.txt'):
		print('file:wordlist.txt does not exist.')
		return 1
	else:
		f = open('wordlist.txt', encoding="utf-8")
		wordlist = f.read().split('\n')
		#print('part of the wordlist is: ',wordlist[0:20])
		f.close()
	if not os.path.isfile(input_filename):
		print('input_file does not exist.')
		return 1
	f = open(input_filename,"r",encoding="utf-8")
	tmp = f.read()
	word_start_index = 0
	one_sequence = []
	counter = [1,1,1]
	temp_word = ''
	#these three numbers are [var,fun,class]
	for j in range(len(tmp)):
		if ((tmp[j] == ' ' or tmp[j] == '\n' or tmp[j] == '\t') and (j != word_start_index)):
			temp_word = tmp[word_start_index:j]
			word_start_index = j+1
			one_sequence.append(temp_word)
			one_sequence.append(tmp[j])
		elif ((tmp[j].isalpha()) or (tmp[j].isdigit())or tmp[j] == '_'):
			pass
		elif (j == len(tmp)-1 and ((tmp[j].isalpha()) or (tmp[j].isdigit()) or tmp[j] == '_')):
			temp_word = tmp[word_start_index:j+1]
			word_start_index = j+1
			one_sequence.append(temp_word)
		elif (tmp[j] in '!"#$%&()*+,-./:;\'<=>?@[\]^`{|}~\\'):
			if j>word_start_index:
				temp_word = tmp[word_start_index:j]
				one_sequence.append(temp_word)
			temp_word = tmp[j]
			word_start_index = j+1
			one_sequence.append(temp_word)
		else:	
			one_sequence.append(tmp[j])
			word_start_index = j+1
	#print('one_sequence:',one_sequence)
	def next_word(i):
		i += 1
		while (one_sequence[i] == ' ' or one_sequence[i] == '\n' or one_sequence[i] == '\t') and (i<len(one_sequence)-1):
			i += 1
		return i
	def last_word(i):
		i -= 1
		while (one_sequence[i] == ' ' or one_sequence[i] == '\n' or one_sequence[i] == '\t') and (i>0):
			i -= 1
		return i
	def replace_all_words(i,counter_index):
		target_word = one_sequence[i]
		type_of_target = ['VAR','FUN','CLASS']
		for j in range(len(one_sequence)):
			if (one_sequence[j] == target_word):
				one_sequence[j] = type_of_target[counter_index]+str(counter[counter_index])
		counter[counter_index] += 1		
	for i in range(len(one_sequence)):
		if ((one_sequence[i][0:3] != 'VAR') and (one_sequence[i][0:3] != 'FUN') and (one_sequence[i][0:5] != 'CLASS') and (is_identifier(one_sequence[i]) == 1) and (one_sequence[i] not in wordlist)):
			if ((i != 0) and (one_sequence[last_word(i)] == 'class')):
				replace_all_words(i,2)
			elif ((i != 0) and (one_sequence[last_word(i)] == 'struct' or one_sequence[last_word(i)] == '}')):
				replace_all_words(i,2)
			elif (i != len(one_sequence)-1) and (one_sequence[next_word(i)] == '('):
				replace_all_words(i,1)
			elif (i>0) and (i<len(one_sequence)-1) and ((one_sequence[next_word(i)] == "'" and one_sequence[last_word(i)] == "'") or (one_sequence[last_word(i)] == '"' and one_sequence[next_word(i)] == '"')):
				pass
			elif ((i != 0) and (one_sequence[last_word(i)] == '%' or one_sequence[last_word(i)] == '\\')):
				pass
			else:
				replace_all_words(i,0)
	write_sequence = "".join(one_sequence)
	f.close()
	return write_sequence



#该函数对指定目录下的漏洞样本进行批量变量名替换处理，将所有程序员定义的变量名，函数名，及类名替换成统一的形式，如VAR1,FUN1等等。
#处理时需要从wordlist文件中读取c/c++的保留字，实现单文件处理的函数需要读一次wordlist，而实际上批量处理所有样本只需要读一次wordlist。
#输入：input_path为输入样本的路径（文件夹），output_path为变量名替换后的样本的输出路径（文件夹）。
def var_fun_transfer_folder(input_path,output_path):
	if not os.path.isfile('wordlist.txt'):
		print('file:wordlist.txt does not exist.')
		return 1
	else:
		f = open('wordlist.txt', encoding="utf-8")
		wordlist = f.read().split('\n')
		print('part of the wordlist is: ',wordlist[0:20])
		f.close()
	if not os.path.isdir(input_path):
		print('input_path:', input_path,' does not exist.')
		return 1
	if not os.path.isdir(output_path):
		os.makedirs(output_path)
		print('output_path does not exist, a new folder has been created.')
	for root, _, files in os.walk(input_path):
		for name in files:
			f = open(os.path.join(root, name), encoding="utf-8")
			tmp = f.read()
			word_start_index = 0
			one_sequence = []
			counter = [1,1,1]
			temp_word = ''
			#these three numbers are [var,fun,class]
			for j in range(len(tmp)):
				if ((tmp[j] == ' ' or tmp[j] == '\n' or tmp[j] == '\t') and (j != word_start_index)):
					temp_word = tmp[word_start_index:j]
					word_start_index = j+1
					one_sequence.append(temp_word)
					one_sequence.append(tmp[j])
				elif ((tmp[j].isalpha()) or (tmp[j].isdigit())or tmp[j] == '_'):
					pass
				elif (j == len(tmp)-1 and ((tmp[j].isalpha()) or (tmp[j].isdigit()) or tmp[j] == '_')):
					temp_word = tmp[word_start_index:j+1]
					word_start_index = j+1
					one_sequence.append(temp_word)
				elif (tmp[j] in '!"#$%&()*+,-./:;\'<=>?@[\]^`{|}~\\'):
					if j>word_start_index:
						temp_word = tmp[word_start_index:j]
						one_sequence.append(temp_word)
					temp_word = tmp[j]
					word_start_index = j+1
					one_sequence.append(temp_word)
				else:	
					one_sequence.append(tmp[j])
					word_start_index = j+1
			#print('one_sequence:',one_sequence)
			def next_word(i):
				i += 1
				while (one_sequence[i] == ' ' or one_sequence[i] == '\n' or one_sequence[i] == '\t') and (i<len(one_sequence)-1):
					i += 1
				return i
			def last_word(i):
				i -= 1
				while (one_sequence[i] == ' ' or one_sequence[i] == '\n' or one_sequence[i] == '\t') and (i>0):
					i -= 1
				return i
			def replace_all_words(i,counter_index):
				target_word = one_sequence[i]
				type_of_target = ['VAR','FUN','CLASS']
				for j in range(len(one_sequence)):
					if (one_sequence[j] == target_word):
						one_sequence[j] = type_of_target[counter_index]+str(counter[counter_index])
				counter[counter_index] += 1		
			for i in range(len(one_sequence)):
				if ((one_sequence[i][0:3] != 'VAR') and (one_sequence[i][0:3] != 'FUN') and (one_sequence[i][0:5] != 'CLASS') and (is_identifier(one_sequence[i]) == 1) and (one_sequence[i] not in wordlist)):
					if ((i != 0) and (one_sequence[last_word(i)] == 'class')):
						replace_all_words(i,2)
					elif ((i != 0) and (one_sequence[last_word(i)] == 'struct' or one_sequence[last_word(i)] == '}')):
						replace_all_words(i,2)
					elif (i != len(one_sequence)-1) and (one_sequence[next_word(i)] == '('):
						replace_all_words(i,1)
					elif (i>0) and (i<len(one_sequence)-1) and ((one_sequence[next_word(i)] == "'" and one_sequence[last_word(i)] == "'") or (one_sequence[last_word(i)] == '"' and one_sequence[next_word(i)] == '"')):
						pass
					elif ((i != 0) and (one_sequence[last_word(i)] == '%' or one_sequence[last_word(i)] == '\\')):
						pass
					else:
						replace_all_words(i,0)
			writefile_name = f.name.split("/")[-1]
			writefile = open(output_path+'/'+writefile_name,"w")
			write_sequence = "".join(one_sequence)
			writefile.write(write_sequence)
			writefile.close()
			f.close()
	print('var_fun_transfer finished.')
	return 0


#该函数对漏洞样本进行大致的分类。根据传入的文件名，根据文件名中的CWE编号给出该样本的大致类别。
#由于CWE分类过于复杂，我们挑选出CWE中样本较多的类别，将它们大致分为10类以及其他。
#输入：name文件名
#输出：返回该样本所属的大致类别编号
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
	elif 'CWE134' in name:
		return 7
	elif ('CWE194' in name) or ('CWE195' in name) or ('CWE196' in name) or ('CWE197' in name):
		return 8
	elif 'CWE457' in name:
		return 9
	elif ('CWE415' in name) or ('CWE416' in name):
		return 10
	else:
		return 11


#该函数用于给出一个漏洞样本的标签。当一个样本是bad类型，则标签为该样本所属类的编号（1-10）。若一个样本无漏洞，则标签为0。
#输入：name为文件名。
#输出：标签。非0标签对应有该类漏洞的样本。0代表无漏洞。
def get_label(name):
	if ('bad' in name):
		return sample_rough_classification(name)
	else:
		return 0


#该函数用于读取数据集，将读取到的数据集和标签作为list返回。
#输入：path为数据集路径。
#输出：返回data，label两个list，分别对应数据和标签。
def get_original_data(path):
	data = []
	label = []
	print('Reading data...')
	for root, _, files in os.walk(path):
		for name in files:
			f = open(os.path.join(root, name), encoding="utf-8")
			tmp = f.read()
			data.append(tmp)
			label.append(get_label(name))
			f.close()
	print('Reading data finished.')
	return data,label


#该函数用于建立一个字典，储存所有在数据集中出现过的词，并将每个词对应一个编号。
#输入：data为数据集，为list形式，list中每一个元素都是一个样本（字符串）
#输出：一个word_index，字典。
def create_dictionary(data):
	print('Creating dictionary...')
	word_index = {}
	#dict_index为字典中下一个词对应的编号。初始值为1，意味着字典中的第一个词编号为1。
	dict_index = 1
	for i in range(len(data)):
		word_start_index = 0
		for j in range(len(data[i])):
			if ((data[i][j].isspace()) and (j != word_start_index)):
				temp_word = data[i][word_start_index:j]
				word_start_index = j+1
				if (temp_word not in word_index):
					word_index[temp_word] = dict_index
					dict_index += 1
			elif ((data[i][j].isalpha()) or (data[i][j].isdigit())or data[i][j] == '_'):
				pass
			elif (j == len(data[i])-1 and ((data[i][j].isalpha()) or (data[i][j].isdigit()) or data[i][j] == '_')):
				temp_word = data[i][word_start_index:j+1]
				word_start_index = j+1
				if (temp_word not in word_index):
					word_index[temp_word] = dict_index
					dict_index += 1
			elif (data[i][j] in '!"#$%&()*+,-./:;\'<=>?@[\]^`{|}~\\'):
				if j>word_start_index:
					temp_word = data[i][word_start_index:j]
					if (temp_word not in word_index):
						word_index[temp_word] = dict_index
						dict_index += 1
				temp_word = data[i][j]
				word_start_index = j+1
				if (temp_word not in word_index):
					word_index[temp_word] = dict_index
					dict_index += 1
			else:
				word_start_index = j+1
	print('Creating dictionary finished.')
	return word_index


#该函数将漏洞代码样本转化成词编号的序列。
#输入：data为数据集，为list形式，list中每一个元素都是一个样本（字符串）,word_index是create_dictionary函数生成的字典。
#输出：sequences为二维list，外层list的每个元素是一个样本对应的词编号序列，里层list每个元素是一个词编号。
def get_sequences(data,word_index):
	print('Getting sequences...')
	sequences = []
	for i in range(len(data)):
		word_start_index = 0
		one_sequence = []
		for j in range(len(data[i])):
			if ((data[i][j].isspace()) and (j != word_start_index)):
				temp_word = data[i][word_start_index:j]
				word_start_index = j+1
				if (temp_word in word_index):
					one_sequence.append(word_index[temp_word])
				else:
					one_sequence.append(0)
			elif ((data[i][j].isalpha()) or (data[i][j].isdigit())or data[i][j] == '_'):
				pass
			elif (j == len(data[i])-1 and ((data[i][j].isalpha()) or (data[i][j].isdigit()) or data[i][j] == '_')):
				temp_word = data[i][word_start_index:j+1]
				word_start_index = j+1
				if (temp_word in word_index):
					one_sequence.append(word_index[temp_word])
				else:
					one_sequence.append(0)
			elif (data[i][j] in '!"#$%&()*+,-./:;\'<=>?@[\]^`{|}~\\'):
				if j>word_start_index:
					temp_word = data[i][word_start_index:j]
					if (temp_word in word_index):
						one_sequence.append(word_index[temp_word])
					else:
						one_sequence.append(0)
				temp_word = data[i][j]
				word_start_index = j+1
				if (temp_word in word_index):
					one_sequence.append(word_index[temp_word])
				else:
					one_sequence.append(0)
			else:
				word_start_index = j+1
		sequences.append(one_sequence)
	print('Getting sequences finished.')
	return sequences


#该函数对上面的词编号序列进行补零操作，将所有小于某一长度的样本补零到该长度，大于该长度的样本舍弃。
#输入：sequences为二维list，外层list每个元素为一个样本，内层list为每个样本中的词编号。label为标签，因为要舍弃部分超长样本，因此label中对应的标签也要舍弃。max_length为补零的长度。
#输出：pad_sequence,pad_label
def pad_zero_to_sequences(sequences,label,max_length):
	print("Start padding...")
	pad_sequence = []
	pad_label = []
	for i in range(len(sequences)):
		if len(sequences[i]) > max_length:
			continue
		w2v = []
		for j in range(len(sequences[i])):
			w2v.append(sequences[i][j])
		while len(w2v) < max_length:
			w2v.append(0)
		pad_sequence.append(w2v)
		pad_label.append(label[i])
	print('Number of pad_sequence:',len(pad_sequence))
	print('Number of pad_label:',len(pad_label))
	print("padding finished.")
	return pad_sequence,pad_label


#该函数将一个数据结构储存到json文件中。
#输入：target为数据，如list，dict等。filename为写入的文件名（字符串）。
#输出：返回0表示执行成功。
def write_data_to_json(target,filename):
	if not '.json' in filename:
		print('Filename must be .json!')
		return 1
	with open(filename,'w',encoding="utf-8") as f:
		json.dump(target,f)
		print('Write target to file:',filename,' finished.')
	return 0


#该函数将一个数据结构从json文件中读取出来。
#输入：filename为文件名。
#输出：将读取出来的数据返回。
def read_data_from_json(filename):
	if not os.path.isfile(filename):
		print('file does not exist.')
		return 1
	with open(filename,'r',encoding="utf-8") as f:
		target = json.load(f)
	print('Read data from file:',filename,' finished.')
	return target


#该函数对待测试的样本进行预处理，包括分词，替换成词编号，补零，得到待测样本的词编号序列。
#输入：filename为文件名，word_index为字典，max_length为最大长度
#输出：返回一个list，为待测样本的词编号序列。
def test_target_preprocess(filename,word_index,max_length):
	f = open(filename,'r',encoding="utf-8")
	tmp = f.read()
	f.close()
	
	#print('Getting test sequence...')
	
	word_start_index = 0
	one_sequence = []
	for j in range(len(tmp)):
		if ((tmp[j].isspace()) and (j != word_start_index)):
			temp_word = tmp[word_start_index:j]
			word_start_index = j+1
			if (temp_word in word_index):
				one_sequence.append(word_index[temp_word])
			else:
				one_sequence.append(0)
		elif ((tmp[j].isalpha()) or (tmp[j].isdigit())or tmp[j] == '_'):
			pass
		elif (j == len(tmp)-1 and ((tmp[j].isalpha()) or (tmp[j].isdigit()) or tmp[j] == '_')):
			temp_word = tmp[word_start_index:j+1]
			word_start_index = j+1
			if (temp_word in word_index):
				one_sequence.append(word_index[temp_word])
			else:
				one_sequence.append(0)
		elif (tmp[j] in '!"#$%&()*+,-./:;\'<=>?@[\]^`{|}~\\'):
			if j>word_start_index:
				temp_word = tmp[word_start_index:j]
				if (temp_word in word_index):
					one_sequence.append(word_index[temp_word])
				else:
					one_sequence.append(0)
			temp_word = tmp[j]
			word_start_index = j+1
			if (temp_word in word_index):
				one_sequence.append(word_index[temp_word])
			else:
				one_sequence.append(0)
		else:
			word_start_index = j+1
	
	#print("target sequence is :",one_sequence)
	
	#print("start target padding")
	if len(one_sequence) > max_length:
		print(len(one_sequence),"is larger than ",max_length,"\n")
		one_sequence = one_sequence[:max_length]
	w2v = []
	for j in range(len(one_sequence)):
		w2v.append(one_sequence[j])
	while len(w2v) < max_length:
		w2v.append(0)
	target_pad_sequence = w2v
	#print('target_pad_sequence is:',target_pad_sequence)
	return target_pad_sequence


#该函数随机从训练集中的各类漏洞中抽取5个，组成测试用的对照样本。按漏洞类别1-10排列成list。
#输入：pad_sequence,pad_label
#输出：test_compare_sample[],list形式。
def create_test_compare_sample(pad_sequence,pad_label):
	test_compare_sample = []
	for kind in range(1,11):
		tmp = []
		for i in range(len(pad_sequence)):
			if pad_label[i] == kind:
				tmp.append(pad_sequence[i])
		print("kind ",kind," has ",len(tmp)," vulnerable samples.")
		for j in range(5):
			test_compare_sample.append(random.choice(tmp))
	print('test_compare_sample has been created.')
	return test_compare_sample














































