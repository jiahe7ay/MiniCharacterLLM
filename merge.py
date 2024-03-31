import json

import os

#data_file_path是数据集存放的文件夹路径
#output_file是合并之后文件的输出路径
def merge_json(data_file_path,output_file):

	files= os.listdir(data_file_path)
	print(files)
	data=[]
	for file in files:
		if "json" not in file:
			continue
		json_data=json.load(open(data_file_path+file,"r"))
		print(json_data[0])
		for cov_data in json_data:
			data.append(cov_data)

	with open(output_file, 'w') as outfile:
		json.dump(data, outfile,indent=4,ensure_ascii=False)


merge_json("./data/", "m_data.json")