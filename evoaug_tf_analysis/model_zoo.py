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


params = {'kernel_size1': 7,
          'kernel_size2': 3,
          'kernel_size3': 5,
          'kernel_size4': 3,
          'num_filters': 256,
          'num_filters2': 60,
          'num_filters3': 60,
          'num_filters4': 120,
          'n_conv_layer': 4,
          'n_add_layer': 2,
          'dropout_prob': 0.4,
          'dense_neurons1': 256,
          'dense_neurons2': 256,
          'pad':'same'}

def DeepSTARR_ori(input_shape, params=params):

    dropout_prob = params['dropout_prob']
    n_conv_layer = params['n_conv_layer']
    n_add_layer = params['n_add_layer']
    
    # body
    inputs = tf.keras.Input(shape=input_shape)
    x = keras.layers.Conv1D(params['num_filters'], kernel_size=params['kernel_size1'],
                  padding=params['pad'],
                  name='Conv1D_1st')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling1D(2)(x)

    for i in range(1, n_conv_layer):
        x = keras.layers.Conv1D(params['num_filters'+str(i+1)],
                      kernel_size=params['kernel_size'+str(i+1)],
                      padding=params['pad'],
                      name=str('Conv1D_'+str(i+1)))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPooling1D(2)(x)
    
    x = keras.layers.Flatten()(x)
    
    # dense layers
    for i in range(0, n_add_layer):
        x = keras.layers.Dense(params['dense_neurons'+str(i+1)],
                     name=str('Dense_'+str(i+1)))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dropout(dropout_prob)(x)
    
    # heads per task (developmental and housekeeping enhancer activities)
    outputs = keras.layers.Dense(2, activation='linear')(x) #tasks = ['Dev', 'Hk']

    return inputs, outputs


def DeepSTARR_v1(input_shape):

    # body
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv1D(256, kernel_size=7, padding='same', kernel_initializer=tf.keras.initializers.HeNormal())(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    #x = keras.layers.LeakyReLU()(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Conv1D(60, kernel_size=3, padding='same', kernel_initializer=tf.keras.initializers.HeNormal())(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    #x = keras.layers.LeakyReLU()(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Conv1D(60, kernel_size=5, padding='same', kernel_initializer=tf.keras.initializers.HeNormal())(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    #x = keras.layers.LeakyReLU()(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Conv1D(120, kernel_size=3, padding='same', kernel_initializer=tf.keras.initializers.HeNormal())(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    #x = keras.layers.LeakyReLU()(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Flatten()(x)

    x = keras.layers.Dense(256)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    #x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dropout(0.4)(x)

    x = keras.layers.Dense(256)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    #x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dropout(0.4)(x)

    outputs = keras.layers.Dense(2, activation='linear')(x)

    #return inputs, outputs
    return tf.keras.Model(inputs=inputs, outputs=outputs)