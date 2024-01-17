import tensorflow as tf
from tensorflow import keras


def DeepSTARR(input_shape, output_shape=2):

    # body
    inputs = tf.keras.Input(shape=input_shape)
    
    x = keras.layers.Conv1D(256, kernel_size=7, padding='same', name='Conv1D_1st')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling1D(2)(x)

    x = keras.layers.Conv1D(60, kernel_size=3, padding='same', name=str('Conv1D_2'))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling1D(2)(x)

    x = keras.layers.Conv1D(60, kernel_size=5, padding='same', name=str('Conv1D_3'))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling1D(2)(x)

    x = keras.layers.Conv1D(120, kernel_size=3, padding='same', name=str('Conv1D_4'))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling1D(2)(x)

    x = keras.layers.Flatten()(x)
    
    # dense layers
    x = keras.layers.Dense(256, name=str('Dense_1'))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.4)(x)
    
    x = keras.layers.Dense(256, name=str('Dense_2'))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.4)(x)
    
    # heads per task (developmental and housekeeping enhancer activities)
    outputs = keras.layers.Dense(output_shape, activation='linear')(x) #tasks = ['Dev', 'Hk']

    return tf.keras.Model(inputs=inputs, outputs=outputs)




def CNN(input_shape):

    # body
    inputs = tf.keras.Input(shape=input_shape)
    
    x = keras.layers.Conv1D(64, kernel_size=7, padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.MaxPooling1D(4)(x)

    x = keras.layers.Conv1D(96, kernel_size=5, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling1D(4)(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Conv1D(128, kernel_size=5, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling1D(5)(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Flatten()(x)
    
    # dense layers
    x = keras.layers.Dense(256)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    
    # heads per task (developmental and housekeeping enhancer activities)
    logits = keras.layers.Dense(1, activation='linear')(x) #tasks = ['Dev', 'Hk']
    outputs = keras.layers.Activation('sigmoid')(logits)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


