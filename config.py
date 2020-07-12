import os

class ConfigParser:
    def parse_config(self, config, mode):
        parsed_config = {}

        # validate settings required for both training and evaluation
        # validate 'data_dir'
        if 'data_dir' not in config:
            raise ValueError("'data_dir' missing from config.")
        if not os.path.isdir(config['data_dir']):
            raise FileNotFoundError("'{0}' is not a valid file path.".format(config['data_dir']))
        parsed_config['data_dir'] = config['data_dir']

        # validate history_length
        if 'history_length' not in config:
            raise ValueError("'history_length' missing from config.")
        if not isinstance(config['history_length'], int):
            raise TypeError("'history_length' must be an integer.")
        parsed_config['history_length'] = int(config['history_length'])

        # validate 'prediction_horizon'
        if 'prediction_horizon' not in config:
            raise ValueError("'prediction_horizon' missing from config.")
        if not isinstance(config['prediction_horizon'], int):
            raise TypeError("'prediction_horizon' must be an integer.")
        parsed_config['prediction_horizon'] = int(config['prediction_horizon'])

        # model config validation
        # validate 'model'
        models = ['linear', 'mlp', 'lstm', 'gru', 'lstm_attention']

        if 'model' not in config:
            raise ValueError("'model' missing from config.")
        if config['model'] not in models:
            raise ValueError("'model' must be one of {0}.".format(models))
        parsed_config['model'] = config['model']

        # validate 'nb_hidden_units'
        if 'nb_hidden_units' not in config:
            raise ValueError("'nb_hidden_units' missing from config.")
        if not isinstance(config['nb_hidden_units'], int):
            raise TypeError("'nb_hidden_units' must be an int.")
        parsed_config['nb_hidden_units'] = int(config['nb_hidden_units'])

        # validate 'nb_layers'
        if config['model'] in ['mlp', 'gru', 'lstm'] and 'nb_layers' not in config:
            raise ValueError("'nb_layers' must be specified when 'model' in {0}.".format(['mlp', 'gru', 'lstm']))
        if not isinstance(config['nb_layers'], int):
            raise TypeError("'nb_layers' must be an int.")
        if 'nb_layers' in config:
            parsed_config['nb_layers'] = int(config['nb_layers'])

        # validate 'nb_attention_units'
        if config['model'] == 'lstm_attention':
            if 'nb_attention_units' not in config:
                raise ValueError("'nb_attention_units' must be specified when 'model' == 'lstm_attention'.")
            if not isinstance(config['nb_attention_units'], int):
                raise TypeError("'nb_attention_units' must be an int.")
            if 'nb_attention_units' in config:
                parsed_config['nb_attention_units'] = int(config['nb_attention_units'])

        # validate 'dropout'
        if config['model'] in ['gru', 'lstm', 'lstm_attention']:
            if 'dropout' in config and not isinstance(config['dropout'], float):
                raise TypeError("'dropout' must be a float.")
            if 'dropout' in config:
                parsed_config['dropout'] = float(config['dropout'])

        # validate 'recurrent_dropout'
        if config['model'] in ['gru', 'lstm', 'lstm_attention']:
            if 'recurrent_dropout' in config and not isinstance(config['recurrent_dropout'], float):
                raise TypeError("'dropout' must be a float.")
            if 'recurrent_dropout' in config:
                parsed_config['recurrent_dropout'] = float(config['recurrent_dropout'])

        # validate 'output'
        if 'output_dir' not in config:
            raise ValueError("'output_dir' missing from config.")
        if not os.path.isdir(config['output_dir']):
            raise FileNotFoundError("'{0}' is not a valid file path.".format(config['output_dir']))
        parsed_config['output'] = config['output_dir']

        # validate settings required for training only
        if mode == 'train':
            # validate 'dataset'
            dataset_options = ['all', 'subject_only', 'exclude_subject']

            if 'dataset' not in config:
                raise ValueError("'dataset' missing from config.")
            if config['dataset'] not in dataset_options:
                raise ValueError("'dataset' must be one of: {0}".format(str(dataset_options)))
            parsed_config['dataset'] = config['dataset']

            # validate 'subject'
            all_subjects = [540, 544, 552, 567, 584, 596, 559, 563, 570, 575, 588, 591]

            if config['dataset'] != 'all' and 'subject' not in config:
                raise ValueError("'subject' must be specified when 'dataset' != 'all'.")
            if config['dataset'] != 'all' and not isinstance(config['subject'], int):
                raise TypeError("'subject' must be an integer.")
            if config['dataset'] != 'all' and config['subject'] not in all_subjects:
                raise ValueError("'subject' must be one of: {0}".format(str(all_subjects)))
            if 'subject' in config['subject']:
                parsed_config['subject'] = int(config['subject'])

            # validate 'train_frac'
            if 'train_frac' not in config:
                raise ValueError("'train_frac' missing from config.")
            if not isinstance(config['train_frac'], float):
                raise TypeError("'train_frac' must be a float.")
            parsed_config['train_frac'] = float(config['train_frac'])

            # validate 'include_missing'
            if 'include_missing' not in config:
                raise ValueError("'include_missing' missing from config.")
            if config['include_missing'] not in [True, False]:
                raise TypeError("'is_missing' must be one of {0}.".format([True, False]))
            parsed_config['include_missing'] = config['include_missing']

            # validate training parameters
            # validate 'batch_size'
            if 'batch_size' not in config:
                raise ValueError("'batch_size' missing from config.")
            if not isinstance(config['batch_size'], int):
                raise TypeError("'batch_size' must be an int.")
            parsed_config['batch_size'] = int(config['batch_size'])

            # validate 'max_epochs'
            if 'max_epochs' not in config:
                raise ValueError("'max_epochs' missing from config.")
            if not isinstance(config['max_epochs'], int):
                raise TypeError("'max_epochs' must be an int.")
            parsed_config['max_epochs'] = int(config['max_epochs'])

            # validate 'nb_runs'
            if 'nb_runs' not in config:
                raise ValueError("'nb_runs' missing from config.")
            if not isinstance(config['nb_runs'], int):
                raise TypeError("'nb_runs' must be an int.")
            parsed_config['nb_runs'] = int(config['nb_runs'])

            # validate 'training_mode'
            training_modes = ['fixed_epochs', 'early_stopping']
            if 'training_mode' not in config:
                raise ValueError("'training_mode' missing from config.")
            if config['training_mode'] not in training_modes:
                raise TypeError("'training_mode' must be one of {0}.".format(training_modes))
            parsed_config['training_mode'] = config['training_mode']

            # validate 'patience'
            if config['training_mode'] == 'early_stopping':
                if 'patience' not in config:
                    raise ValueError("'patience' must be specified when 'training_mode' == 'early_stopping'.")
                if not isinstance(config['patience'], int):
                    raise TypeError("'patience' must be an int.")
                if 'patience' in config:
                    parsed_config['patience'] = int(config['patience'])

        # validate settings required for evaluation only
        if mode == 'evaluate':
            # validate 'data_mean'
            if 'data_mean' not in config:
                raise ValueError("'data_mean' missing from config.")
            if not isinstance(config['data_mean'], float):
                raise TypeError("'data_mean' must be a float.")
            parsed_config['data_mean'] = float(config['data_mean'])

            # validate 'data_stdev'
            if 'data_stdev' not in config:
                raise ValueError("'data_stdev' missing from config.")
            if not isinstance(config['data_stdev'], float):
                raise TypeError("'data_stdev' must be a float.")
            parsed_config['data_stdev'] = float(config['data_stdev'])

            # validate 'model_fp'
            if 'model_fp' not in config:
                raise ValueError("'model_fp' missing from config.")
            if not os.path.isfile(config['model_fp']):
                raise FileNotFoundError("'{0}' is not a valid file path.".format(config['model_fp']))
            parsed_config['model_fp'] = config['model_fp']

        return parsed_config