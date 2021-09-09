train_data_dir = "trainingData"
test_data_dir = "testData"
original_data_dir = "data/HAM10000_images_part_1"
original_test_data_dir = "data/HAM10000_images_part_2"

##### HYPERPARAMETERS
# NN_IMAGE_COUNT = sample_count_in_folder(train_data_dir)
NN_BATCH_SIZE = 2
NN_CLASSIFIER_WIDTH = 256
NN_EPOCH_COUNT = 200
NN_FINETUNE_EPOCHS = 200
NN_VALIDATION_PROPORTION = 0.2
NN_TARGET_WIDTH = 224
NN_TARGET_HEIGHT = 224
NN_LEARNING_RATE = 1e-6
