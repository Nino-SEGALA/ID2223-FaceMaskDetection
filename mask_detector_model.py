
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras import backend as K

from sklearn.model_selection import train_test_split
from os import listdir
from tqdm import tqdm


# Limiting GPU memory growth
# https://www.tensorflow.org/guide/gpu
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# create the network model
def create_model():
    # Note the input shape is the 224x224 images with 3 channels for RGB
    inputs = Input((224,224,3))

    # 1st set of Convolutional and maxpool layer
    conv1 = Conv2D(filters=32,kernel_size=3,strides=2,activation='relu', kernel_initializer='he_normal',
                   padding='same')(inputs)
    pool1 = MaxPool2D((2, 2))(conv1)

    # 2st set of Convolutional and maxpool layer with 20% Dropout as a regulizer
    drop1 = Dropout(0.2)(pool1)
    conv2 = Conv2D(filters=64, kernel_size=3, strides=2, activation='relu', kernel_initializer='he_normal',
                   padding='same')(drop1)
    pool2 = MaxPool2D((2, 2))(conv2)

    # 3d set of Convolutional layer with 20% Dropout as a regulizer, however this time maxpool is not used,
    # since the output of the convolutional layer is the prime number 7
    drop2 = Dropout(0.2)(pool2)
    conv3 = Conv2D(filters=128, kernel_size=3, strides=2, activation='relu', kernel_initializer='he_normal',
                   padding='same')(drop2)

    # Flatten the resulting Matrix to 1D array
    flat1 = Flatten()(conv3) # BYT TILL POOL4

    # Hidden layer with 128 nodes and ReLU activation function Tried 512 as well
    dens1 = Dense(128, activation='relu')(flat1) # add a fully connected layer in the end

    # Output layer with single node since it's a binary classifier and therefore also sigmoid is used
    output = Dense(1, activation='sigmoid')(dens1)

    model = Model(inputs=[inputs], outputs=[output])

    # 1e-4 i standard and did from our test work the best
    model.compile(optimizer = Adam(lr=1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    plot_model(model, to_file='mask_classifier.png', show_shapes=True)
    return model


def plots(history):
    #https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    #plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.tight_layout()

    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    #plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.tight_layout()
    plt.show()


# credit https://stackoverflow.com/questions/45466020/how-to-export-keras-h5-to-tensorflow-pb
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


directory = 'Data'
X, Y = list(), list()
for i, folder in enumerate(listdir(directory)):

    new = directory+'/'+folder
    for image in tqdm(listdir(new)):
        filename = new + '/' + image
        image = load_img(filename, target_size=(224, 224))

        image = img_to_array(image)

        X.append(image)
        Y.append(i)


#reshape data and put it in random order and split into train, test and validation set
train_Y = np.array(Y).reshape(-1, 1)
train_X = np.array(X)
train_X, test_X, train_Y, test_Y = train_test_split(train_X, train_Y,test_size=0.10, random_state=45) # maybe change size?
train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y,test_size=0.11, random_state=45) # maybe change size

# create a data generator
augmentation = ImageDataGenerator(
        rotation_range=30,
        shear_range=0.2,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode="nearest",
        vertical_flip=True,
        horizontal_flip=True)

# hyper params
n_epochs = 90
batch_size = 32


model = create_model()
callbacks = [EarlyStopping(patience=10,
                           monitor='val_loss')]

history = model.fit(augmentation.flow(x=train_X,y=train_Y, batch_size=batch_size),
                    steps_per_epoch=len(train_X) / batch_size,
                    epochs=n_epochs,
                    validation_data=(val_X,val_Y),
                    callbacks=callbacks)


results = model.evaluate(x=test_X, y=test_Y)
print("test loss, test acc:", results)

# plot accuracy and loss for training and validation set
plots(history)

# save the model
model.save('Model/my_model.hdf5')


frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])

tf.train.write_graph(frozen_graph, "Model", "my_model.pb", as_text=False)


# some experiments and optimization notes

# Epoch 26/60 , 512 nodes in dense layer
#test loss, test acc: [0.5202921852469444, 0.8151041865348816]
#dict_keys(['val_loss', 'val_accuracy', 'loss', 'accuracy'])

# 128 nodes in dense layer
#test loss, test acc: [0.40253619352976483, 0.875]
#dict_keys(['val_loss', 'val_accuracy', 'loss', 'accuracy'])

# end at 7 nodes instead
#test loss, test acc: [0.1913989099363486, 0.9557291865348816]
#dict_keys(['val_loss', 'val_accuracy', 'loss', 'accuracy'])

# higher learning rate
#test loss, test acc: [0.25476594393452007, 0.9322916865348816]
#dict_keys(['val_loss', 'val_accuracy', 'loss', 'accuracy'])
