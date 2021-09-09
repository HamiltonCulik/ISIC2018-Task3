from sklearn.utils import class_weight
import config
import plotting

import numpy as np
import tensorflow as tf


from keras import layers, models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict


from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.python.keras.preprocessing.image import DirectoryIterator

# converts y_truths according to map given in target_label #TODO finish implementing this
class SelectiveGenerator(DirectoryIterator):

	def __init__(self, target_label, mode, data_generator):
		self.target_label = target_label
		super().__init__(
			directory=data_generator.directory,
			image_data_generator=data_generator.image_data_generator,
			target_size=data_generator.target_size,
			color_mode=data_generator.color_mode,
			classes=None,
			class_mode=data_generator.class_mode,
			data_format=data_generator.image_data_generator.data_format,
			batch_size=data_generator.batch_size,
			shuffle=data_generator.shuffle,
			seed=data_generator.seed,
			save_to_dir=None,
			save_prefix="",
			save_format="png",
			follow_links=False,
			subset=data_generator.subset,
			interpolation="nearest")

		if(mode == "binary"):
			self.num_classes = 2
			self.classes = list(map(lambda x: 1 if(x == np.argmax(
				self.target_label)) else 0, data_generator.classes))
			self.class_mode = "binary"

			# class_names = self.class_indices.keys()
			# self.class_indices = dict(zip(class_names, [1 if (i==np.argmax(self.target_label)) else 0 for i in range(len(class_names))])))

	# def _set_index_array(self):
	# 	self.index_array = np.arange(len(self.classes))
	# 	# falses = self.classes.count(0)

	# 	if self.shuffle:
	# 		trues = self.classes.count(1)
	# 		self.n = trues * 2
	# 		aux = map(lambda x: x if self.classes[x] == 1 else -1, self.index.array)
	# 		aux = list(filter(lambda x: x != -1, aux) ) #trues indexes
	# 		aux2 = map(lambda x: x if self.classes[x] == 0 else -1, self.index.array)
	# 		aux2 = list(filter(lambda x: x != -1, aux)) # falses indexes
	# 		np.random.shuffle(aux2)
	# 		equalized_set = np.array(aux[:self.classes.count(1)] + aux[self.classes.count(1):self.classes.count(1)*2])
	# 		print(equalized_set)
	# 		np.random.shuffle(equalized_set)
	# 		print(equalized_set)
	# 		self.index_array = equalized_set

	# def _get_batches_of_transformed_samples(self, index_array):
	# 	batch_x, batch_y = super()._get_batches_of_transformed_samples(index_array)
	# 	if(self.mode == "binary"):
	# 		print(batch_y)
	# 		print(target_label)
	# 		batch_y = [[1] if(np.argmax(label)==np.argmax(self.target_label)) else [0] for label in batch_y]
	# 	elif(self.mode == "categorical"):
	# 		#TODO target_label is a map. check highest class number used, reshape to that, then one hot encode every batch according to their original class and their new mapped class
	# 		pass

	# 	return batch_x, batch_y


def export_labels(generator):
	# path = None
	cont = 0
	aux = []
	for data_batch, labels_batch in generator:

		print(labels_batch)
		# if (not path):
		# path = data_batch[0].split("/")
		# path = path[:-3]
		# path = path.join()
		for label in labels_batch:
			# np.append(aux, label)
			print(label)
			# aux.append(label.argmax())
		cont += 1
		if cont > len(generator.classes)/config.NN_BATCH_SIZE:
			break
	return aux


def calculate_class_weights(generator, mu=0.15):
	class_weights_dict = dict()
	for label in generator.classes:
		if(class_weights_dict.get(label) == None):
			class_weights_dict[label] = 0
		class_weights_dict[label] += 1

	print("Class proportions: " + str([class_weights_dict[label] /
	      len(generator.classes) for label in class_weights_dict.keys()]))
	for key in class_weights_dict:
	    class_weights_dict[key] = (1/class_weights_dict[key]) * (len(generator.classes) / generator.num_classes)  # sanity check
	# score = np.log(mu*len(generator.classes)/float(class_weights_dict[key])) #TODO check if normalized weights are better
	# class_weights_dict[key] = score if score > 1 else 1
	return class_weights_dict


def train_model(model, patience=8, loss="categorical_crossentropy", acc_mode="accuracy", monitor="val_loss", output_shape=(None, 7), target_label=np.array([])):
	train_datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2, zoom_range=0.2, horizontal_flip=True, rotation_range=90,
                                    validation_split=config.NN_VALIDATION_PROPORTION)  # generator also augments data for us

	validation_datagen = ImageDataGenerator(rescale=1./255,
                                         validation_split=config.NN_VALIDATION_PROPORTION)  # we create a different generator for validation so it doesn't augment data

	random_seed = np.random.randint(0, 10000)
	train_generator = train_datagen.flow_from_directory(
		config.train_data_dir,
		target_size=(config.NN_TARGET_WIDTH, config.NN_TARGET_HEIGHT),
		batch_size=config.NN_BATCH_SIZE,
		subset="training",
		seed=random_seed,
		class_mode="categorical")  # then we use the same seed in both, so that they end up with the same images selected as validation
	validation_generator = validation_datagen.flow_from_directory(
		config.train_data_dir,
		target_size=(config.NN_TARGET_WIDTH, config.NN_TARGET_HEIGHT),
		batch_size=config.NN_BATCH_SIZE,
		subset="validation",
		seed=random_seed,
		class_mode="categorical")

	sample_count_training = len(train_generator.classes)
	sample_count_validation = len(validation_generator.classes)

	if(target_label.any()):
		aux_train_generator = train_generator
		aux_validation_generator = validation_generator
		train_generator = SelectiveGenerator(
			target_label=target_label, mode="binary", data_generator=aux_train_generator)
		validation_generator = SelectiveGenerator(
			target_label=target_label, mode="binary", data_generator=aux_validation_generator)

	train_dataset = tf.data.Dataset.range(8).interleave(
		lambda _:  tf.data.Dataset.from_generator(
                    lambda: train_generator,
                    output_signature=(tf.TensorSpec(shape=(None, config.NN_TARGET_WIDTH, config.NN_TARGET_HEIGHT, 3)),
                                      tf.TensorSpec(shape=output_shape))  # output shape is an argument, useful when trainiing binary classification
                ).prefetch(tf.data.AUTOTUNE),
		num_parallel_calls=tf.data.AUTOTUNE
	)

	val_dataset = tf.data.Dataset.range(8).interleave(
		lambda _:  tf.data.Dataset.from_generator(
                    lambda: validation_generator,
                    output_signature=(tf.TensorSpec(shape=(None, config.NN_TARGET_WIDTH, config.NN_TARGET_HEIGHT, 3)),
                                      tf.TensorSpec(shape=output_shape))
                ).prefetch(tf.data.AUTOTUNE),
		num_parallel_calls=tf.data.AUTOTUNE
	)

	class_weight_dict = class_weight.compute_class_weight("balanced",
                                                       classes=np.unique(
                                                           train_generator.classes),
                                                       y=train_generator.classes)
	class_weight_dict = {i: value for i, value in enumerate(class_weight_dict)}
	print(class_weight_dict)
	print(calculate_class_weights(train_generator))  # sanity check

	model.summary()
	#TODO experiment with SGD and aggressive regularization and decay
	model.compile(loss=loss,
               optimizer=optimizers.Adam(lr=config.NN_LEARNING_RATE),
               # optimizer=optimizers.SGD(lr=config.NN_LEARNING_RATE, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=1),
               metrics=[acc_mode])

	if(monitor != "val_loss" and monitor != "val_binary_crossentropy"):
		mode = "max"
	else:
		mode = "min"
	Callbacks = [EarlyStopping(patience=patience, restore_best_weights=False, monitor="loss", mode="min", verbose=1),
              #  ReduceLROnPlateau(patience=2),
              ModelCheckpoint(f"{model.name}.h5",
                              save_best_only=True,
                              monitor=monitor, mode=mode, verbose=1)]

	history = model.fit(
            x=train_dataset,
         			epochs=config.NN_EPOCH_COUNT,
         			steps_per_epoch=(
                                    sample_count_training//config.NN_BATCH_SIZE),
         			validation_data=val_dataset,
         			validation_steps=(
                                    (sample_count_validation-1)//config.NN_BATCH_SIZE),
         			class_weight=class_weight_dict,
         			callbacks=Callbacks,
         			verbose=1,
	)

	return history, model


def test_model(test_data_dir, modelname, target_label=np.array([])):
	model = models.load_model(modelname)
	test_datagen = ImageDataGenerator(rescale=1./255)

	test_generator = test_datagen.flow_from_directory(
		test_data_dir,
		target_size=(config.NN_TARGET_WIDTH, config.NN_TARGET_HEIGHT),
		batch_size=config.NN_BATCH_SIZE,
		shuffle=False)
	if(target_label.any()):
		test_generator = SelectiveGenerator(
			target_label=target_label, mode="binary", data_generator=test_generator)
	Y_pred = model.predict(x=test_generator, steps=len(
		test_generator.classes)//config.NN_BATCH_SIZE+1)

	if(target_label.any()):
		target_names = ["False", "True"]
		y_pred = [[1] if (pred > 0.5) else [0] for pred in Y_pred]

	else:
		import os
		target_names = sorted(os.listdir(test_data_dir))
		y_pred = np.argmax(Y_pred, axis=1)

	print(len(test_generator.classes))
	print('Confusion Matrix')
	matrix_result = confusion_matrix(test_generator.classes, y_pred)
	print(matrix_result)
	print('Classification Report')
	print(classification_report(test_generator.classes,
	      y_pred, target_names=target_names))

	plotting.plot_confusion_matrix(matrix_result, target_names)
