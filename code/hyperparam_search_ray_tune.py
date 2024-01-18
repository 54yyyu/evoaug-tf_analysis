import os, sys, h5py
from filelock import FileLock
import wandb
wandb.login()
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
import evoaug_tf
from evoaug_tf import evoaug_tf, augment, model_zoo
import utils

import ray
ray.init(ignore_reinit_error=True)
from ray import air,tune
from ray.air import session
from pathlib import Path
from ray.tune.integration.keras import TuneReportCallback
#from ray.tune.search.hyperopt import HyperOptSearch
#from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.schedulers import PopulationBasedTraining
from ray.air.integrations.wandb import WandbLoggerCallback

RESULTS_DIR = '../result_ray/'
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

config = {
            "max_augs_per_seq": tune.choice([1,2,3,4]), 
            "rc_prob": tune.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5]),
            "shift_max": tune.choice([0, 10, 20, 30, 40, 50]),
            "insert_max": tune.choice([0, 10, 20, 30, 40, 50]), 
            "delete_max": tune.choice([0, 10, 20, 30, 40, 50]), 
            "noise_std": tune.choice([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]), 
            "mutate_frac": tune.choice([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]),
        }

def train_evoaug(config):

    batch_size = 100
    epochs = 100

    filepath = '../data/DeepSTARR_data.h5'

    with FileLock(os.path.expanduser("~/.data.lock")):
        x_train, y_train, x_valid, y_valid, x_test, y_test = utils.H5DataLoader(filepath)

    _, L, A = x_valid.shape 

    augment_lists = [
        augment.RandomDeletion(delete_min=0, delete_max=config["delete_max"]),
        augment.RandomRC(rc_prob=config["rc_prob"]), 
        augment.RandomInsertion(insert_min=0, insert_max=config["insert_max"]), 
        augment.RandomTranslocation(shift_min=0, shift_max=config["shift_max"]), 
        augment.RandomNoise(noise_mean=0, noise_std=config["noise_std"]),
        augment.RandomMutation(mutate_frac=config["mutate_frac"])
    ]
    
    augment_list = []
    
    if config["delete_max"]:
        augment_list.append(augment_lists[0])
    if config["rc_prob"]:
        augment_list.append(augment_lists[1])
    if config["insert_max"]:
        augment_list.append(augment_lists[2])
    if config["shift_max"]:
        augment_list.append(augment_lists[3])
    if config["noise_std"]:
        augment_list.append(augment_lists[4])
    if config["mutate_frac"]:
        augment_list.append(augment_lists[5])

    model = evoaug_tf.RobustModel(model_zoo.DeepSTARR, input_shape=(L,A), augment_list=augment_list, max_augs_per_seq=config["max_augs_per_seq"], hard_aug=True)
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer, loss='mse') 
                
    # early stopping callback
    es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', #'val_aupr',#
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
    
    ckpt_path = os.path.join(RESULTS_DIR, f'del{config["delete_max"]}_rc{config["rc_prob"]}_ins{config["insert_max"]}_tra{config["shift_max"]}_noi{config["noise_std"]}_mut{config["mutate_frac"]}_aug.h5')
    checkpoint = ModelCheckpoint(filepath=ckpt_path,
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
                    callbacks=[es_callback, reduce_lr, checkpoint, TuneReportCallback({"loss": "loss", "val_loss":'val_loss'})])

    model.save_weights(f'{RESULTS_DIR}del{config["delete_max"]}_rc{config["rc_prob"]}_ins{config["insert_max"]}_tra{config["shift_max"]}_noi{config["noise_std"]}_mut{config["mutate_frac"]}_aug.h5')


def tune_evoaug(num_training_iterations, num_samples):
    # sched = AsyncHyperBandScheduler(
    #     time_attr="training_iteration", max_t=100, grace_period=20
    # )
    sched = PopulationBasedTraining(
        time_attr="training_iteration", 
        perturbation_interval=2,
        metric="val_loss",
        mode="min", 
        hyperparam_mutations=config, 
    )

    tuner = tune.Tuner(
        tune.with_resources(train_evoaug, resources={"cpu": 4, "gpu": 1}),
        tune_config=tune.TuneConfig(
            #search_alg = HyperOptSearch(),
            #metric="val_loss",
            #mode="min",
            scheduler=sched,
            num_samples=num_samples,
        ),
        run_config=air.RunConfig(
            name="evoaug_hyperparam_search_pbt",
            stop={"training_iteration": num_training_iterations},
            callbacks=[WandbLoggerCallback(project="evoaug")]
        ),
        param_space=config,
    )
    results = tuner.fit()
    result_df = results.get_dataframe()
    result_df.to_csv(RESULTS_DIR + 'tune_all.csv')



class ModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, monitor='val_loss', mode='min', save_best_only=True, **kwargs):
        super(ModelCheckpoint, self).__init__(filepath, monitor=monitor, mode=mode, save_best_only=save_best_only, **kwargs)
        self.best_val_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get(self.monitor)
        if val_loss is None:
            return

        if self.monitor_op(val_loss, self.best_val_loss):
            self.best_val_loss = val_loss
            self.model.save_weights(self.filepath)
            #print(f"Saved model weights at epoch {epoch+1} with validation loss: {val_loss:.4f}")
            



if __name__ == "__main__":
    
    tune_evoaug(num_training_iterations=100,num_samples=200)
    ray.shutdown()
