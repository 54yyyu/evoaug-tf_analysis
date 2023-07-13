import os, sys
import wandb
wandb.login()
from wandb.keras import WandbCallback
import numpy as np
from six.moves import cPickle
import tensorflow as tf
from tensorflow import keras
from evoaug_tf import evoaug_tf, augment, model_zoo
from evoaug_tf.utils import pearson_r, Spearman, H5DataLoader, evaluate_model

group_name = "analysis"

filepath = '../datasets/deepstarr_data.h5'
x_train, y_train, x_valid, y_valid, x_test, y_test = H5DataLoader(filepath)
_, L, A = x_valid.shape

output_dir = 'result/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

augment_lists = [
    augment.RandomDeletion(delete_min=0, delete_max=30),
    augment.RandomRC(rc_prob=0.5),
    augment.RandomInsertion(insert_min=0, insert_max=20),
    augment.RandomTranslocation(shift_min=0, shift_max=20),
    augment.RandomNoise(noise_mean=0, noise_std=0.3),
    augment.RandomMutation(mutate_frac=0.05)
]
exp_name_list = ["deletion", "rc", "insertion", "translocation", "noise", "mutation"]

num_trials = 5

for aug_index in range(len(augment_lists)):
    augment_list = [augment_lists[aug_index]]
    expt_name = exp_name_list[aug_index]

    trial_aug_results = []
    trial_finetune_results = []
    
    exp_dir = output_dir+expt_name+"/"
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        
    for trial in range(num_trials):
        exp_name = expt_name+"_"+str(trial)

        keras.backend.clear_session()

        batch_size = 100
        epochs = 100
        finetune_epochs = 5

        try:
            wandb.finish()
        except:
            pass
        wandb.init(project="evoaug", group=group_name, name=exp_name)

        inputs, outputs = model_zoo.DeepSTARR(input_shape=(L+evoaug_tf.augment_max_len(augment_list),A))#, tasks=['Dev','Hk'])
        model = evoaug_tf.RobustModel(inputs=inputs, outputs=outputs, augment_list=augment_list, max_augs_per_seq=2, hard_aug=True)

        model.compile(keras.optimizers.Adam(learning_rate=0.001, weight_decay=1e-6), #weight_decay
                    loss='mse', #  ['mse','mse']
                    #loss_weights=[1, 1], # loss weigths to balance
                    metrics=[Spearman, pearson_r]) # additional track metric

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

        ckpt_path = os.path.join(exp_dir, exp_name+"_aug.h5")
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path,
                                            monitor='val_loss',
                                            save_best_only=True,
                                            mode = 'min',
                                            save_freq='epoch',)


        # train model
        model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(x_valid, y_valid),
                        callbacks=[es_callback, reduce_lr, checkpoint, WandbCallback()])

        model.load_weights(ckpt_path)

        pred = model.predict(x_test, batch_size=64)
        aug_results = evaluate_model(y_test, pred)

        logs = {
            'test_pearson_r (Dev)': aug_results[0][0],
            'test_pearson_r (Hk)': aug_results[0][1],
            'test_Spearman (Dev)': aug_results[1][0],
            'test_Spearman (Hk)': aug_results[1][1],
        }
        wandb.log(logs)

        # ----------------- Fine Tune Analysis -----------------

        finetune_optimizer = keras.optimizers.Adam(learning_rate=0.0001, weight_decay=1e-6)
        model.finetune_mode(optimizer=finetune_optimizer)

        ckpt_path = os.path.join(exp_dir, exp_name+"_finetune.h5")
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path,
                                            monitor='val_loss',
                                            save_best_only=True,
                                            mode = 'min',
                                            save_freq='epoch',)

        # train model
        model.fit(x_train, y_train,
                        epochs=finetune_epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(x_valid, y_valid),
                        callbacks=[checkpoint]) #WandbCallback()

        model.load_weights(ckpt_path)

        pred = model.predict(x_test, batch_size=64)
        finetune_results = evaluate_model(y_test, pred)

        logs = {
            'FT_test_pearson_r (Dev)': finetune_results[0][0],
            'FT_test_pearson_r (Hk)': finetune_results[0][1],
            'FT_test_Spearman (Dev)': finetune_results[1][0],
            'FT_test_Spearman (Hk)': finetune_results[1][1],
        }
        wandb.log(logs)

        # store results
        trial_aug_results.append(aug_results)
        trial_finetune_results.append(finetune_results)

        wandb.finish()

    with open(os.path.join(output_dir, expt_name+'_analysis.pickle'), 'wb') as fout:
        cPickle.dump(trial_finetune_results, fout)
        cPickle.dump(trial_aug_results, fout)