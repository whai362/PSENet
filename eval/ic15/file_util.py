import os

def read_dir(root):
	file_path_list = []
	for file_path, dirs, files in os.walk(root):
		for file in files:
			file_path_list.append(os.path.join(file_path, file).replace('\\', '/'))
	file_path_list.sort()
	return file_path_list

def read_file(file_path):
	file_object = open(file_path, 'r')
	file_content = file_object.read()
	file_object.close()
	return file_content

def write_file(file_path, file_content):
	if file_path.find('/') != -1:
		father_dir = '/'.join(file_path.split('/')[0:-1])
		if not os.path.exists(father_dir):
			os.makedirs(father_dir)
	file_object = open(file_path, 'w')
	file_object.write(file_content)
	file_object.close()


def write_file_not_cover(file_path, file_content):
	father_dir = '/'.join(file_path.split('/')[0:-1])
	if not os.path.exists(father_dir):
		os.makedirs(father_dir)
	file_object = open(file_path, 'a')
	file_object.write(file_content)
	file_object.close()