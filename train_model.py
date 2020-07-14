import json
import sys
import os
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping

import data
import models
from utils import RMSE
from config import ConfigParser

# validate input
try:
    config_fp = sys.argv[1]
except IndexError:
    print("You failed to provide a configuration file. Usage: {0} config.json".format(sys.argv[0]))

if '.json' not in sys.argv[1]:
    raise ValueError("The configuration file must be in JSON format.")

# parse config
config = json.load(open(config_fp, 'r'))

# parse config
config = ConfigParser().parse_config(config=config, mode='train')

# load data
exclude = []

all_subjects = [540, 544, 552, 567, 584, 596, 559, 563, 570, 575, 588, 591]

if config['dataset'] == 'subject_only':
    exclude = list(set(all_subjects) - set([int(config['subject'])]))
elif config['dataset'] == 'exclude_subject':
    exclude = [int(config['subject'])]

print("loading data..")
X_train, Y_train, X_val, Y_val, X_train_mean, X_train_stdev = data.load_data(dir=config['data_dir'], exclude=exclude,
            history_length=config['history_length'], prediction_horizon=config['prediction_horizon'],
            include_missing=config['include_missing'], train_frac=config['train_frac'])
print("data loaded.\nmean = {0}\tstdev = {1}".format(round(X_train_mean, 3), round(X_train_stdev, 3)))

if config['model'] in ['gru', 'lstm', 'lstm_attention']:
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

# load model
if config['model'] == 'linear':
    model = models.LinearModel(input_shape=(config['history_length'],), nb_output_units=1)
elif config['model'] == 'mlp':
    model = models.MLPModel(input_shape=(config['history_length'],), nb_output_units=1,
                            nb_hidden_units=config['nb_hidden_units'], nb_layers=config['nb_layers'])
elif config['model'] == 'gru':
    model = models.GRUModel(input_shape=(config['history_length'], 1), nb_output_units=1,
                            nb_hidden_units=config['nb_hidden_units'], nb_layers=config['nb_layers'],
                            dropout=config['dropout'], recurrent_dropout=config['recurrent_dropout'])
elif config['model'] == 'lstm':
    model = models.LSTMModel(input_shape=(config['history_length'], 1), nb_output_units=1,
                            nb_hidden_units=config['nb_hidden_units'], nb_layers=config['nb_layers'],
                            dropout=config['dropout'], recurrent_dropout=config['recurrent_dropout'])
elif config['model'] == 'lstm_attention':
    model = models.LSTMAttentionModel(input_shape=(config['history_length'], 1), nb_output_units=1,
                             nb_hidden_units=config['nb_hidden_units'], dropout=config['dropout'],
                             recurrent_dropout=config['recurrent_dropout'], nb_attention_units=config['nb_attention_units'])

# model training
loss_function = RMSE

for ridx in range(config['nb_runs']):
    print("Run #{0}".format(ridx+1))

    # build & compile model
    m = model.build()
    m.compile(loss=RMSE,
                  optimizer='adam',
                  metrics=[RMSE])

    callbacks = []

    if config['training_mode'] == 'early_stopping':
        callbacks.append(ModelCheckpoint(filepath='{0}/{1}_{2}.pkl'.format(config['output'], str(model), ridx+1),
                                         save_best_only=True, save_weights_only=True))
        callbacks.append(EarlyStopping(patience=config['patience']))


    # train model
    hist = m.fit(X_train, Y_train,
                     batch_size=config['batch_size'],
                     epochs=config['max_epochs'],
                     shuffle=True,
                     validation_data=(X_val, Y_val),
                     callbacks=callbacks
                 )

    if config['training_mode'] == 'fixed_epochs':
        print("Saving model to '{0}/{1}_{2}.pkl'".format(config['output'], str(model), ridx+1))
        m.save_weights('{0}/{1}_{2}.pkl'.format(config['output'], str(model), ridx+1))
    elif config['training_mode'] == 'early_stopping':
        print("Loading model weights: '{0}/{1}_{2}.pkl'".format(config['output'], str(model), ridx+1))
        m.load_weights('{0}/{1}_{2}.pkl'.format(config['output'], str(model), ridx+1))

    train_loss = X_train_stdev * m.evaluate(X_train, Y_train)[1]
    val_loss = X_train_stdev * m.evaluate(X_val, Y_val)[1]

    print("training RMSE = {0}".format(train_loss))
    print("validation RMSE = {0}".format(val_loss))
