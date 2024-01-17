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
    augment_list.append(augment.RandomInsertionBatch(insert_min=0, insert_max=20))
if 'T' in aug_list:
    augment_list.append(augment.RandomTranslocationBatch(shift_min=0, shift_max=20))
if 'D' in aug_list:
    augment_list.append(augment.RandomDeletionBatch(delete_min=0, delete_max=20))
if 'M' in aug_list:
    augment_list.append(augment.RandomMutation(mutate_frac=0.05))
if 'N' in aug_list:
    augment_list.append(augment.RandomNoise(noise_mean=0, noise_std=0.2))

# other parameters
output_dir = '../results_chip/'
num_trials = 5
batch_size = 100
max_epochs = 100
max_augs_per_seq = 1
finetune_epochs = 10


exp_names = ['ATF2', 'BACH1', 'CTCF', 'ELK1', 'GABPA', 'MAX', 'REST', 'SRF', 'ZNF24']

for exp in exp_names:

    expt_name = 'chip_' + aug_list + '_' + exp

    # load eclip dataset
    filepath = '../data/chip/'+exp+'_200.h5'
    x_train, y_train, x_valid, y_valid, x_test, y_test = utils.H5DataLoader(filepath, lower_case=True, transpose=True, downsample=None)
    _, L, A = x_valid.shape

    # loop over trials
    trial_aug_results = []
    trial_finetune_results = []
    trial_standard_results = []
    for trial in range(num_trials):

        # load evoaug model
        model = evoaug.RobustModel(model_zoo.CNN, (L,A), augment_list=augment_list, max_augs_per_seq=1, hard_aug=True)
        aug_optimizer = keras.optimizers.Adam(learning_rate=0.001)
        loss = keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.0)
        auroc = keras.metrics.AUC(curve='ROC', name='auroc')
        model.compile(aug_optimizer, loss=loss, metrics=[auroc])

        # early stopping callback
        es_callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=10,
                                                    verbose=1,
                                                    mode='min',
                                                    restore_best_weights=True)
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

        model.save_weights(os.path.join(output_dir, expt_name+"_aug_"+str(trial)))

        # evaluate model
        scores = model.evaluate(x_test, y_test, batch_size=512)
        trial_aug_results.append(scores[1])


        # ----------------- Fine Tune Analysis -----------------

        model.finetune_mode(lr=0.0001)
        es_callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=3,
                                                    verbose=1,
                                                    mode='min',
                                                    restore_best_weights=True)

        # finetune model
        model.fit(x_train, y_train,
                    epochs=finetune_epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(x_valid, y_valid),
                    callbacks=[es_callback])

        model.save_weights(os.path.join(output_dir, expt_name+"_finetune_"+str(trial)))

        # evaluate model
        scores = model.evaluate(x_test, y_test, batch_size=512)
        trial_finetune_results.append(scores[1])

        # ----------------- Standard Analysis -----------------

        # load evoaug model
        model = model_zoo.CNN(input_shape=(L,A))
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        loss = keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.0)
        auroc = keras.metrics.AUC(curve='ROC', name='auroc')
        model.compile(optimizer, loss=loss, metrics=[auroc])

        # early stopping callback
        es_callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=10,
                                                    verbose=1,
                                                    mode='min',
                                                    restore_best_weights=True)

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
        scores = model.evaluate(x_test, y_test, batch_size=512)
        trial_standard_results.append(scores[1])
        

        # clear graph
        keras.backend.clear_session()
        gc.collect()

    # save results
    with open(os.path.join(output_dir, expt_name+'_results.pickle'), 'wb') as fout:
        cPickle.dump(trial_aug_results, fout)
        cPickle.dump(trial_finetune_results, fout)
        cPickle.dump(trial_standard_results, fout)

