import os, sys, gc
import numpy as np
from six.moves import cPickle
from tensorflow import keras
import utils, model_zoo

# other parameters
expt_name = 'deepstarr'
output_dir = '../result/'
num_trials = 5
batch_size = 100
max_epochs = 100

# load deepstarr dataset
filepath = '../data/DeepSTARR_data.h5'
x_train, y_train, x_valid, y_valid, x_test, y_test = utils.H5DataLoader(filepath)
_, L, A = x_valid.shape

# loop over trials
trial_standard = []
for trial in range(num_trials):

    # load evoaug model
    model = model_zoo.DeepSTARR(input_shape=(L,A))
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer, loss='mse')

    # early stopping callback
    es_callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=10,
                                                verbose=1,
                                                mode='min',
                                                restore_best_weights=True)
    # reduce learning rate callback
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                    factor=0.1, #0.2,
                                                    patience=5, #3,
                                                    min_lr=1e-7,
                                                    mode='min',
                                                    verbose=1)

    # train model
    model.fit(x_train, y_train,
                epochs=max_epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_valid, y_valid),
                callbacks=[es_callback, reduce_lr])

    model.save_weights(os.path.join(output_dir, expt_name+"_standard_"+str(trial)))

    # evaluate model
    pred = model.predict(x_test, batch_size=512)
    results = utils.evaluate_model(y_test, pred)
    trial_standard.append(results)

    # clear graph
    keras.backend.clear_session()
    gc.collect()

# save results
with open(os.path.join(output_dir, expt_name+'_standard_results.pickle'), 'wb') as fout:
    cPickle.dump(trial_standard, fout)



