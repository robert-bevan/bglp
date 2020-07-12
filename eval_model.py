import json
import sys
import os
import numpy as np

import data
import models
from utils import RMSE, MAE
from config import ConfigParser

# REMOVE THIS
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
config = ConfigParser().parse_config(config=config, mode='evaluate')

# load model
if config['model'] == 'linear':
    model = models.LinearModel(input_shape=(config['history_length'],), nb_output_units=1,
                               nb_hidden_units=config['nb_hidden_units'])
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
                                      recurrent_dropout=config['recurrent_dropout'],
                                      nb_attention_units=config['nb_attention_units'])

# build & compile model
model_str = str(model)
model = model.build()

loss_function = RMSE
model.compile(loss=RMSE,
          optimizer='adam',
          metrics=[RMSE])

print("Loading model weights: '{0}'".format(config['model_fp']))
model.load_weights(config['model_fp'])

# evaluate model
subjects = [540, 544, 552, 567, 584, 596]

res = {}

for subject_index in subjects:
    X_test, Y_test, test_times = data.load_test_data(dir=config['data_dir'], subject_index=subject_index,
                                                     history_length=config['history_length'],
                                                     prediction_horizon=config['prediction_horizon'],
                                                     data_mean=config['data_mean'], data_stdev=config['data_stdev'])

    if config['model'] in ['gru', 'lstm', 'lstm_attention']:
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # evaluate model
    rmse = config['data_stdev']*model.evaluate(X_test, Y_test)[1]

    preds = [(p[0] * config['data_stdev']) + config['data_mean'] for p in model.predict(X_test)]
    mae = MAE(np.array([y*config['data_stdev'] + config['data_mean'] for y in Y_test]), np.array(preds))

    res[subject_index] = {'MAE':mae, 'RMSE':rmse}

    # write predictions to file
    with open("{0}/{1}_{2}_{3}".format(config['output'], model_str, subject_index, config['prediction_horizon'] * 5), 'w') as f:
        [f.write("{0} {1}\n".format(t, round(preds[idx], 2))) for idx, t in enumerate(test_times)]

print("{0}\nPerformance Summary\n{1}".format("="*50, "="*50))
for subject_index in res:
    print("{0}\nRMSE = {1}\nMAE = {2}\n".format(subject_index, round(res[subject_index]['RMSE'], 3),
                                                round(res[subject_index]['MAE'], 3)))

