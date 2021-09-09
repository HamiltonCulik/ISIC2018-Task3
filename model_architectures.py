import config

from keras import layers, models
from tensorflow.keras import regularizers

#TODO regularization on layers prone to overfitting when using SGD

def lens_net(focus): #simple convolutional binary classifier model. Output is sigmoidal probability of classification 

    input = layers.Input(shape=(config.NN_TARGET_WIDTH, config.NN_TARGET_HEIGHT, 3), name=f"input_{focus}")
    conv1 = layers.Conv2D(16, kernel_size=(3, 3), name=f"conv1_{focus}",
        activation='relu', strides=(1, 1))(input)
    norm1 = layers.BatchNormalization(name=f"normalization1_{focus}")(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2), name=f"maxpool1_{focus}")(norm1)

    padding1 = layers.ZeroPadding2D(padding=(1, 1), name=f"padding1_{focus}")(pool1)
    conv2 = layers.Conv2D(32, kernel_size=(3, 3), name=f"conv2_{focus}",
        activation='relu', strides=(1, 1))(padding1)
    norm2 = layers.BatchNormalization(name=f"normalization2_{focus}",)(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2), name=f"maxpool2_{focus}")(norm2)
    padding2 = layers.ZeroPadding2D(padding=(1, 1), name=f"padding2_{focus}")(pool2)
    conv3 = layers.Conv2D(64, kernel_size=(3, 3), name=f"conv3_{focus}",
        activation='relu', strides=(1, 1))(padding2)
    norm3 = layers.BatchNormalization(name=f"normalization3_{focus}")(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2), name=f"maxpool3_{focus}")(norm3)
    padding3 = layers.ZeroPadding2D(padding=(1, 1), name=f"padding3_{focus}")(pool3)
    conv4 = layers.Conv2D(64, kernel_size=(3, 3), name=f"conv4_{focus}",
        activation='relu', strides=(1, 1))(padding3)
    norm4 = layers.BatchNormalization(name=f"normalization4_{focus}")(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2), name=f"maxpool4_{focus}")(norm4)
    padding4 = layers.ZeroPadding2D(padding=(1, 1), name=f"padding4_{focus}")(pool4)
    conv5 = layers.Conv2D(128, kernel_size=(3, 3), name=f"conv5_{focus}",
                          activation='relu', strides=(1, 1))(padding3)
    flatten = layers.Flatten(name=f"flatten_{focus}")(conv5)
    dense1 = layers.Dense(32, activation='relu', name=f"dense1_{focus}")(flatten)
    dropout1 = layers.Dropout(0.5, name=f"dropout1_{focus}")(dense1)
    dense2 = layers.Dense(32, activation='relu', name=f"dense2_{focus}")(dropout1)
    dropout2 = layers.Dropout(0.5, name=f"dropout2_{focus}")(dense2)
    output = layers.Dense(1, activation='sigmoid', name=f"output_{focus}")(dropout2)

    return models.Model(inputs=input, outputs=output, name=f"{focus}_{config.NN_TARGET_WIDTH}_{config.NN_LEARNING_RATE}")


def multi_lens_net(): #7 embedded models result in competition and better accuracy for underrepresented classes
    import os
    target_names = sorted(os.listdir(config.train_data_dir))
    this_dir = os.listdir()


    input = layers.Input(shape=(config.NN_TARGET_WIDTH, config.NN_TARGET_HEIGHT, 3))
    focusing_lens = [
        models.load_model(f"{focus}_{config.NN_TARGET_WIDTH}_{config.NN_LEARNING_RATE}.h5")(input) if (f"{focus}_{config.NN_TARGET_WIDTH}_{config.NN_LEARNING_RATE}.h5" in this_dir)
        else lens_net(focus)(input)
        for focus in target_names
    ]

    # for lens in focusing_lens:
    # 	lens.trainable = False

    sections_output = layers.Concatenate()(focusing_lens) #Output is a concatenation of every binary sigmoidal prediction for every embedded model


    #TODO A classifier that played around with sigmoidal probabilities has so far worsened prediction rates
    # final_classifier = layers.Dense(32, activation='relu')(sections_output)
    # dropout = layers.Dropout(0.4)(final_classifier)
    # final_output = layers.Dense(7, activation="softmax")(sections_output)

    #TODO also did not yet implement a proportional normalization for class weights in output. Might improve accuracy? 
    # proportionality_coefficient = tf.convert_to_tensor(list(class_weight_dict.values()))
    # def proportionalize(x):
    # 	aux = [1 if i == tf.argmax(tf.multiply(
    # 		x, proportionality_coefficient)).numpy else 0 for i in range(7)]
    # 	tf.keras.backend.set_value(x, [None,aux])
    # final_output = layers.Lambda(lambda x: proportionalize(x))(sections_output)
    # final_output = np.zeros(7)
    # final_output[np.argmax(np.multiply(sections_output, proportionality_coefficient))] = 1
    # final_output = layers.Maximum()(sections_output)

    model = models.Model(inputs=input, outputs=sections_output, name=f"mln_{config.NN_TARGET_WIDTH}_{config.NN_LEARNING_RATE}")
    return model

def mln_final(): #Increase learning rate when training only dense classifiers, but not too much

    #Main idea was to extract features acquired from the last convolution in every embedded model, but so far this has given unsatisfactory results 
    cnn_base = models.load_model(f"mln_{config.NN_TARGET_WIDTH}_{config.NN_LEARNING_RATE}.h5")
    cnn_base.trainable = False
    # cnn_base.summary()

    feature_models = [model for model in cnn_base.layers[1:-1]]

    feature_extraction_models = [models.Model(
        inputs=model.input, outputs=model.layers[-6].output) for model in feature_models] #layer -6 is flattened results of last convolution

    final_input = layers.Input(shape=(config.NN_TARGET_WIDTH, config.NN_TARGET_HEIGHT, 3))
    concatenation = layers.Concatenate()([feature_extractor(final_input) for feature_extractor in feature_extraction_models])
    dense_classifier = layers.Dense(128, activation="relu", name=f"classifier")(concatenation) 
    dropout = layers.Dropout(0.4)(dense_classifier)
    dense_classifier2 = layers.Dense(128, activation="relu", name=f"classifier2")(dropout)
    dropout2 = layers.Dropout(0.4)(dense_classifier2)
    dense_classifier3 = layers.Dense(64, activation="relu", name=f"classifier2")(dropout)
    dropout3 = layers.Dropout(0.4)(dense_classifier3)
    final_output = layers.Dense(7, activation="softmax", name="final_output")(dropout3)

    model = models.Model(inputs=final_input,outputs=final_output, name="mln_final")
    
    return model
