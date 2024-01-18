import os, sys, gc
import numpy as np
from six.moves import cPickle
from tensorflow import keras
from evoaug_tf import evoaug, augment
import utils, model_zoo

aug_list = sys.argv[1]

augment_list = []
if 'R' in aug_list:
    augment_list.append(augment.RandomRC(rc_prob=0.5))
if 'I' in aug_list:
    augment_list.append(augment.RandomInsertion(insert_min=0, insert_max=20))
if 'T' in aug_list:
    augment_list.append(augment.RandomTranslocation(shift_min=0, shift_max=20))
if 'D' in aug_list:
    augment_list.append(augment.RandomDeletion(delete_min=0, delete_max=30))
if 'M' in aug_list:
    augment_list.append(augment.RandomMutation(mutate_frac=0.05))
if 'N' in aug_list:
    augment_list.append(augment.RandomNoise(noise_mean=0, noise_std=0.3))

# other parameters
expt_name = 'deepstarr_' + aug_list
output_dir = '../results/'
num_trials = 5
batch_size = 100
max_epochs = 100
max_augs_per_seq = 2
finetune_epochs = 10

# load deepstarr dataset
filepath = '../data/DeepSTARR_data.h5'
x_train, y_train, x_valid, y_valid, x_test, y_test = utils.H5DataLoader(filepath)
_, L, A = x_valid.shape

# loop over trials
trial_aug_results = []
trial_finetune_results = []    
for trial in range(num_trials):

    # load evoaug model
    model = evoaug.RobustModel(model_zoo.DeepSTARR, (L,A), augment_list=augment_list, max_augs_per_seq=max_augs_per_seq, hard_aug=True)
    aug_optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(aug_optimizer, loss='mse')

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

    aug_path = os.path.join(output_dir, expt_name+"_aug_"+str(trial))
    model.save_weights(aug_path)

    # evaluate model
    pred = model.predict(x_test, batch_size=512)
    aug_results = utils.evaluate_model(y_test, pred)
    trial_aug_results.append(aug_results)


    # ----------------- Fine Tune Analysis -----------------

    model = evoaug.RobustModel(model_zoo.DeepSTARR, (L,A), augment_list=augment_list, max_augs_per_seq=max_augs_per_seq, hard_aug=True)
    finetune_optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(finetune_optimizer, loss='mse')
    model.finetune_mode(optimizer=finetune_optimizer)
    model.load_weights(aug_path)

    es_callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=5,
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

    # finetune model
    model.fit(x_train, y_train,
              epochs=finetune_epochs,
              batch_size=batch_size,
              shuffle=True,
              validation_data=(x_valid, y_valid),
              callbacks=[es_callback, reduce_lr]) 

    finetune_path = os.path.join(output_dir, expt_name+"_finetune_"+str(trial))
    model.save_weights(finetune_path)

    # evaluate model
    pred = model.predict(x_test, batch_size=512)
    finetune_results = utils.evaluate_model(y_test, pred)
    trial_finetune_results.append(finetune_results)

    # clear graph
    keras.backend.clear_session()
    gc.collect()

# save results
with open(os.path.join(output_dir, expt_name+'_results.pickle'), 'wb') as fout:
    cPickle.dump(trial_finetune_results, fout)
    cPickle.dump(trial_aug_results, fout)



