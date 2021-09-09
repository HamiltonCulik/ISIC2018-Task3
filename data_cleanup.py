import pandas as pd

def organize_data_in_folders(original_data_dir, destination_dir):
	metadata = "data/HAM10000_metadata.csv"
	df = pd.read_csv(metadata)

	import os
	import shutil
	tmp = os.listdir(original_data_dir)

	data_list = []
	for i in tmp:
		data_list.append( (i, df[df["image_id"]==i[:-4]]["dx"].values[0]) ) #append several tuples containing a file name and their respective class

	data_list.sort(key=lambda x: x[0]) #order by filename
	# labels_ordered = np.array([data[1] for data in data_list]) #labels have to be ordered before being used by the generators, so that they will correctly pair with their respective images in generation

	if(not os.path.exists(destination_dir)):
		os.mkdir(destination_dir)

	for data in data_list:
		src = original_data_dir + "/" + data[0]
		if(not os.path.exists(destination_dir + "/" + data[1])):
			os.mkdir(destination_dir + "/" + data[1])
		dst = destination_dir + "/" + data[1] + "/" + data[0]
		shutil.copyfile(src, dst)

def sample_count_in_folder(data_dir):
	import os
	tmp = os.listdir(data_dir)

	count = 0
	for folder in tmp:
		count += len(os.listdir(data_dir + "/" + folder))

	return count

def move_samples(source, destination, proportion):
	import os
	import shutil
	from random import shuffle
	tmp = os.listdir(source)
	count = 0
	for folder in tmp:
		samples = os.listdir(source+"/"+folder)
		shuffle(samples)
		for sample in samples[:int(len(samples)*proportion)]:
			src = source + "/" + folder + "/" + sample
			dst = destination + "/" + folder + "/" + sample
			shutil.copyfile(src, dst)
			os.remove(src)
			count += 1
	print("Moving {} files to new destination".format(count))

def move_samples_countable(source, destination, total):
	import os
	import shutil
	from random import shuffle
	tmp = os.listdir(source)
	for folder in tmp:
		count = 1
		samples = os.listdir(source+"/"+folder)
		shuffle(samples)
		for sample in samples:
			src = source + "/" + folder + "/" + sample
			dst = destination + "/" + folder + "/" + sample
			shutil.copyfile(src, dst)
			os.remove(src)
			count += 1
			if(count > total or count > len(samples)):
				break
	print("Moving {} files to new destination".format(count))

