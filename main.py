# Data comes from https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000
import numpy as np

import config
import model_architectures
import plotting
import testing_routines


# organize_data_in_folders(original_data_dir,train_data_dir)
# organize_data_in_folders(original_test_data_dir, test_data_dir)
# move_samples(test_data_dir, train_data_dir, 0.1)
# move_samples(train_data_dir, test_data_dir, 1.0)
# move_samples_countable(test_data_dir, train_data_dir, 100)
# move_samples_countable(train_data_dir, test_data_dir, 10000)

if __name__ == "__main__":
	print("\n\n\n\n\n\n\n\n\n\n\n")
	# print("Total samples for training = " + str(sample_count_in_folder(train_data_dir)))
	# print("Total samples for testing = " + str(sample_count_in_folder(test_data_dir)))

	import os
	target_names = sorted(os.listdir(config.train_data_dir))
	for name in target_names:
		target_label = np.zeros(len(target_names))
		target_label[target_names.index(name)] = 1
		# history, trained_model = testing_routines.train_model(
		# 	model = model_architectures.lens_net(name),
		# 	patience=4,
		# 	loss="BinaryCrossentropy",
		# 	output_shape = (config.NN_BATCH_SIZE,),
		# 	target_label = target_label
		# )
		# test_model(config.test_data_dir,f"sequential_{name}_{config.NN_TARGET_WIDTH}_{config.NN_LEARNING_RATE}.h5", target_label=target_label)

	# testing_routines.test_model(config.test_data_dir, f"mln_{config.NN_TARGET_WIDTH}_{config.NN_LEARNING_RATE}.h5")
	# testing_routines.test_model(config.test_data_dir, "mln_final.h5")
	model = model_architectures.multi_lens_net()

	# from keras.utils import plot_model
	# plot_model(model, "modelgraphshapes.png", show_layer_names = False, show_shapes=True)

	history, trained_model = testing_routines.train_model(model, 
	patience = 12, 
	monitor = "val_categorical_accuracy", 
	acc_mode="categorical_accuracy"
	)
	plotting.showPlots(history, acc_mode="categorical_accuracy")


