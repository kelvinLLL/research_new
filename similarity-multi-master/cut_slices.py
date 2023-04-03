# -*- coding: utf-8 -*-  

import sys
import re
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

# 不用考虑的C/C++关键词列表
wordlist = [
    'asm',
    'do',
    'if',
    'return',
    'typedef',
    'auto',
    'double',
    'inline',
    'short',
    'typeid',
    'bool',
    'dynamic_cast',
    'int',
    'signed',
    'typename',
    'break',
    'else',
    'long',
    'sizeof',
    'union',
    'case',
    'enum',
    'mutable',
    'static',
    'unsigned',
    'catch',
    'explicit',
    'namespace',
    'static_cast',
    'using',
    'char',
    'export',
    'new',
    'struct',
    'virtual',
    'class',
    'extern',
    'operator',
    'switch',
    'void',
    'const',
    'false',
    'private',
    'template',
    'volatile',
    'const_cast',
    'float',
    'protected',
    'this',
    'wchar_t',
    'continue',
    'for',
    'public',
    'throw',
    'while',
    'default',
    'friend',
    'register',
    'true',
    'delete',
    'goto',
    'reinterpret_cast',
    'try',
    'alignas',
    'alignof',
    'char16_t',
    'char32_t',
    'constexpr',
    'decltype',
    'noexcept',
    'nullptr',
    'static_assert',
    'thread_local'
]

# 敏感函数
sensitive_funcs = [
    'fgetchar',
    'getch',
    'putch',
    'getchar',
    'putchar',
    'getche',
    'ungetch',
    'cgets',
    'scanf',
    'vscanf',
    'cscanf',
    'sscanf',
    'vsscanf',
    'puts',
    'cputs',
    'printf',
    'vprintf',
    'cprintf',
    'vcprintf',
    'sprintf',
    'vsprintf',
    'read',
    'getc',
    'putc',
    'getw',
    'putw',
    'ungetc',
    'fgetc',
    'fputc',
    'fgets',
    'fputs',
    'fread',
    'fwrite',
    'fscanf',
    'vfscanf',
    'fprintf',
    'vfprintf',
    'fseek',
    'ftell',
    'setbuf',
    'setvbuf',
    'getdfree',
    'getdtasetdta',
    'getfat',
    'getfatd',
    'memccpy',
    'memchr',
    'memcmp',
    'memicmp',
    'memmove',
    'memcpy',
    'memset',
    'movedata',
    'movemem',
    'setmem',
    'stpcpy',
    'strcat',
    'strchr',
    'strcmp',
    'strcpy',
    'strcspn',
    'strdup',
    'stricmp',
    'strlen',
    'strlwr',
    'strncat',
    'strncmp',
    'strncpy',
    'strnicmp',
    'strnset',
    'strpbrk',
    'strrchr',
    'strrev',
    'strset',
    'strspn',
    'strstr',
    'strtok',
    'strupr',
    'aloocmem',
    'freemem',
    'coreleft',
    'calloc',
    'malloc',
    'free',
    'realloc',
    'farcalloc',
    'farmalloc',
    'farfree',
    'farrealloc'
]

def main(argv):

    global functions
    global func_starts
    global func_ends
    global text
    global file_len
    global file_list

    # 提示输入格式
    if len(argv) != 2 and len(argv) != 4:
        print("input format: python cut_slices.py filelist [file_name] [line_number]")
        return
    elif len(argv) == 4:
        sensitive_file_name = argv[2]
        sensitive_file_line = int(argv[3])
    else:
        sensitive_file_line = -1


    # 读取文件列表
    with open(argv[1], 'r', encoding='utf-8') as f:
        file_list = f.readlines()

    # 按行读取文件，存储在text中
    text = []
    file_len = []
    for file_name in file_list:
        fn = file_name.strip()
        try:
            with open(fn, 'r', encoding='utf-8') as f:
                tmp = f.readlines()
        except:
            continue
        text.extend(tmp)
        file_len.append(len(tmp))
    text_len = len(text)

    if text_len == 0:
        exit()

    # 记录切片的索引信息
    sensitive_line_to_func = {}
    index_info = []

    # 标记敏感函数所在行号
    sensitive_lines = []
    

    if sensitive_file_line >= 0:
        new_line_num = 0
        for i in range(len(file_list)):
            if file_list[i].strip() == sensitive_file_name:
                new_line_num += sensitive_file_line
                break
            else:
                new_line_num += file_len[i]
        sensitive_lines.append(new_line_num)
        sensitive_line_to_func[new_line_num] = 'unknown'
    else:
        exist_sensitive_funcs = set()
        in_comment = False
        for i in range(text_len):
            line = text[i].split('//')[0]
            if '*/' in line:
                in_comment = False
                continue
            if re.match(r'/\*', line.strip()) is not None:
                in_comment = True
            if in_comment:
                continue
            
            for func in sensitive_funcs:
                pattern = r'[\s,=\(.:]' + func + r'\s*\('
                if re.search(pattern, line) is not None:
                    sensitive_lines.append(i)
                    sensitive_line_to_func[i] = func
                    file_name = get_real_file_name(i)
                    exist_sensitive_funcs.add((file_name, func))
                    break
        print_sensitive_funcs = []
        for (file_name, func) in exist_sensitive_funcs:
            print_sensitive_funcs.append(file_name + '\t' + func + '\n')
        # with open('slicing_criterion.txt', 'w') as f:
        #     f.writelines(print_sensitive_funcs)

    # 标记函数定义起止的行号
    pattern = r'([\w&*]+\s+)*(\w+)\s*\('
    functions = []
    func_starts = []
    func_ends = []
    big_brace = -1
    for i in range(text_len):
        if len(text[i].strip()) > 0 and text[i].strip()[0] == '}':
            big_brace = i
        function = re.match(pattern, text[i])
        if function is not None:
            # print(function.group(0))
            if len(functions) > 0:
                func_ends.append(big_brace)
            functions.append(function.group(2))
            func_starts.append(i)
    if len(functions) > 0:
        func_ends.append(big_brace)
    func_starts.append(text_len)

    # 开始切片
    print("cutting slices of sensitive functions...")
    index = 0
    for sen_line in sensitive_lines:
        func_index = get_func_index(sen_line)
        if func_index is None:
            continue
        index += 1
        stack = []
        word_used = []
        print_line_map = [0] * text_len
        print_line_map[sen_line] = 1
        find_other_words(sen_line, stack, word_used, func_index)

        # 寻找所有调用该函数的其他函数
        func_stack = [func_index]
        func_used = []
        while func_stack:
            fi = func_stack.pop(-1)
            func_used.append(fi)
            for i in range(text_len):
                new_fi = get_func_index(i)
                if new_fi is None:
                    continue
                function = re.search(r'[\s,=\(.:]' + functions[fi] + r'\s*\(', text[i])
                if function is not None:
                    print_line_map[i] = 1
                    find_other_words(i, stack, word_used, new_fi)
                    if new_fi not in func_stack and new_fi not in func_used:
                        func_stack.append(new_fi)

        # print(stack)
        while stack:
            word, word_type, func_index = stack.pop(-1).split('/')
            # print(stack)
            func_index = int(func_index)
            if word_type == 'func':
                find_other_words(func_starts[func_index], stack, word_used, func_index)
                print_line_map[func_starts[func_index]] = 1
            else:
                func_start_line = func_starts[func_index]
                func_end_line = func_ends[func_index]
                in_comment = False
                for line_num in range(func_start_line, func_end_line + 1):
                    line = text[line_num].split('//')[0]
                    if '*/' in line:
                        in_comment = False
                        continue
                    if re.match(r'/\*', line.strip()) is not None:
                        in_comment = True
                    if in_comment:
                        continue

                    line_words = re.split(r'\n| |\t|\(|\)|,|=|;|\.|/|\{|\}|\[|\]|!|<|>|~|&|\||^|:|\+|-|\*', line)
                    for line_word in line_words:
                        if word == line_word:
                            find_other_words(line_num, stack, word_used, func_index)
                            print_line_map[line_num] = 1
                            break

        print_lines = []
        print_line_num = ''
        for line_num in range(text_len):
            if print_line_map[line_num] == 1:
                print_line_num += get_real_line_num(line_num) + ','
                print_lines.append(text[line_num])
        print_line_num = print_line_num.rstrip(',')

        # print(print_lines)
        fout = 'slice' + str(index) + '.txt'
        with open(fout, 'w', encoding='utf-8') as f:
            f.writelines(print_lines)

        sen_func = sensitive_line_to_func[sen_line]
        index_info.append('{:^5d}   {:^18s}   {:^14d}   {}\n'.format(index, sen_func, sen_line, print_line_num))

    # print(index_info)
    with open('index.txt', 'w', encoding='utf-8') as f:
        f.write('index   sensitive function   sensitive line   slice line\n')
        f.writelines(index_info)

    print('cut slice completed.')


# 在某一行中，一个词与之前从stack里取出的词（keyword）在同一行里，即视为与keyword相关，需要继续寻找与这个词相关的词，也就是说要把他压入stack
# 该函数即找出某一行中与keyword相关的所有词并压入stack
# 这个做法只是权宜之计，并不合理，并不是在同一行里就一定相关，需要细化改进
def find_other_words(line_num, stack, word_used, func_index):
    # 获取该行的内容
    line = text[line_num]

    # 匹配函数调用
    it = re.finditer(r'[\s,=\(.:](\w+)\s*\(', line)
    funcs = []
    for match in it:
        func_name = match.group(1)
        if func_name in functions:
            index = functions.index(func_name)
            func_format = func_name + '/func/' + str(index)
            if func_format not in word_used and func_format not in stack:
                stack.append(func_format)
                word_used.append(func_format)
        funcs.append(func_name)

    # 分词
    words = re.split(r'\n| |\t|\(|\)|,|=|;|\.|/|\{|\}|\[|\]|!|<|>|~|&|\||^|:|\+|-|\*', line)
    for word in words:
        if word is not None and word != '' and word not in wordlist and \
          word not in sensitive_funcs and word not in funcs and is_identifier(word) == 1:
            word_format = word + '/var/' + str(func_index)
            if word_format not in word_used and word_format not in stack:
                stack.append(word_format)
                word_used.append(word_format)

    # print(stack)
    # print(word_used)


# 判断一个字符串是否符合identifier的要求，第一个字符不能为数字
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


# 获取所在函数的范围
def get_func_index(line_num):
    i = 0
    while i < len(func_starts) and func_starts[i] <= line_num:
        i += 1
    if i == 0 or func_ends[i-1] < line_num:
        return
    return i - 1


# 获取行号所在的文件名以及真实行号
def get_real_line_num(line_num):
    for i in range(len(file_len)):
        if line_num >= file_len[i]:
            line_num -= file_len[i]
        else:
            break
    return file_list[i].strip() + ':' + str(line_num)


# 获取行号所在的文件名
def get_real_file_name(line_num):
    for i in range(len(file_len)):
        if line_num >= file_len[i]:
            line_num -= file_len[i]
        else:
            break
    return file_list[i].strip()



if __name__ == '__main__':
    main(sys.argv)

