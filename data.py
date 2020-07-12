import xml.etree.ElementTree as ET
import datetime
import numpy as np

MISSING = 'MISSING'
NB_TEST_STEPS_TO_SKIP = 12
FIVE_MIN_SECONDS = 300

def load_data(dir, exclude=[], history_length=6, prediction_horizon=6, include_missing=True, train_frac=0.8):
    X_trains, Y_trains, X_vals, Y_vals = [], [], [], []

    old_sub_indices = [559, 563, 570, 575, 588, 591]
    new_sub_indices = [540, 544, 552, 567, 584, 596]

    all_vals = []
    all_vals_train = []

    for subject_index in old_sub_indices:
        if subject_index not in exclude:
            # read train directory examples
            tree = ET.parse(dir + '/OhioT1DM-training/' + str(subject_index) + '-ws-training.xml')
            root = tree.getroot()
            glucose_level_els = [e for e in root.find('glucose_level').findall('event')]
            glucose_level_time_dict_train = dict(
                [(datetime.datetime.strptime(el.attrib['ts'], '%d-%m-%Y %H:%M:%S'), float(el.attrib['value'])) for el in
                 glucose_level_els])

            # fill in missing values
            times = list(glucose_level_time_dict_train.keys())

            glucose_level_dict_include_missing_train = {}

            for idx, t in enumerate(times):
                if idx > 0:
                    diff = (times[idx] - times[idx - 1]).seconds
                    if diff > FIVE_MIN_SECONDS + 10:
                        # add missing values
                        for jdx in range(int((diff / FIVE_MIN_SECONDS) - 1)):
                            temp = times[idx - 1] + datetime.timedelta(seconds=(jdx + 1) * FIVE_MIN_SECONDS)
                            glucose_level_dict_include_missing_train[temp] = 'MISSING'

                    # add current value
                    glucose_level_dict_include_missing_train[t] = glucose_level_time_dict_train[t]
                else:
                    glucose_level_dict_include_missing_train[t] = glucose_level_time_dict_train[t]

            # read test directory examples
            tree = ET.parse(dir + '/OhioT1DM-testing/' + str(subject_index) + '-ws-testing.xml')
            root = tree.getroot()
            glucose_level_els = [e for e in root.find('glucose_level').findall('event')]
            glucose_level_time_dict_test = dict(
                [(datetime.datetime.strptime(el.attrib['ts'], '%d-%m-%Y %H:%M:%S'), float(el.attrib['value'])) for el in
                 glucose_level_els])

            times = list(glucose_level_time_dict_test.keys())

            glucose_level_dict_include_missing_test = {}

            for idx, t in enumerate(times):
                if idx > 0:
                    diff = (times[idx] - times[idx - 1]).seconds
                    if diff > FIVE_MIN_SECONDS + 10:
                        # add missing values
                        for jdx in range(int((diff / FIVE_MIN_SECONDS) - 1)):
                            temp = times[idx - 1] + datetime.timedelta(seconds=(jdx + 1) * FIVE_MIN_SECONDS)
                            glucose_level_dict_include_missing_test[temp] = 'MISSING'

                    # add current value
                    glucose_level_dict_include_missing_test[t] = glucose_level_time_dict_test[t]
                else:
                    glucose_level_dict_include_missing_test[t] = glucose_level_time_dict_test[t]

            glucose_level_time_dict = {**glucose_level_dict_include_missing_train, **glucose_level_dict_include_missing_test}
            glucose_times = list(glucose_level_time_dict.values())

            all_vals += [x for x in glucose_times if x != 'MISSING']

            nb_train_ex = int(train_frac * len(glucose_times))
            glucose_vals_train = glucose_times[:nb_train_ex]
            all_vals_train += [x for x in glucose_vals_train if x != 'MISSING']

            # validation examples should start an hour after start of validation set
            glucose_vals_val = glucose_times[nb_train_ex + NB_TEST_STEPS_TO_SKIP:]

            X_train, Y_train = [], []
            for idx in range(len(glucose_vals_train) - (history_length + prediction_horizon)):
                # discard examples where Y = MISSING, or all X values = MISSING
                if (glucose_vals_train[idx + history_length + prediction_horizon - 1] != MISSING):
                    if include_missing:
                        if glucose_vals_train[idx:idx + history_length].count(MISSING) != len(glucose_vals_train[idx:idx + history_length]):
                            X_train.append(glucose_vals_train[idx:idx + history_length])
                            Y_train.append(glucose_vals_train[idx + history_length + prediction_horizon - 1])
                    else:
                        if MISSING not in glucose_vals_train[idx:idx + history_length]:
                            X_train.append(glucose_vals_train[idx:idx + history_length])
                            Y_train.append(glucose_vals_train[idx + history_length + prediction_horizon - 1])

            X_val, Y_val = [], []
            for idx in range(len(glucose_vals_val) - (history_length + prediction_horizon)):
                # discard examples where Y = MISSING, or all X values = MISSING
                if (glucose_vals_val[idx + history_length + prediction_horizon - 1] != MISSING):
                    if include_missing:
                        if glucose_vals_val[idx:idx + history_length].count(MISSING) != len(glucose_vals_val[idx:idx + history_length]):
                            X_val.append(glucose_vals_val[idx:idx + history_length])
                            Y_val.append(glucose_vals_val[idx + history_length + prediction_horizon - 1])
                    else:
                        if MISSING not in glucose_vals_val[idx:idx + history_length]:
                            X_val.append(glucose_vals_val[idx:idx + history_length])
                            Y_val.append(glucose_vals_val[idx + history_length + prediction_horizon - 1])

            # add first examples in validation set
            nb_additional_ex = history_length + prediction_horizon - 1
            for jdx in range(nb_additional_ex):
                # discard examples where Y = MISSING, or all X values = MISSING
                if (glucose_vals_val[jdx] != MISSING):
                    temp = glucose_vals_train + glucose_vals_val
                    temp_sidx = len(glucose_vals_train) - (history_length + prediction_horizon - 1) + jdx

                    if include_missing:
                        if temp[temp_sidx:temp_sidx + history_length].count(MISSING) != len(temp[temp_sidx:temp_sidx + history_length]):
                            X_val.append(temp[temp_sidx:temp_sidx + history_length])
                            Y_val.append(glucose_vals_val[jdx])
                    else:
                        if MISSING not in temp[temp_sidx:temp_sidx + history_length]:
                            X_val.append(temp[temp_sidx:temp_sidx + history_length])
                            Y_val.append(glucose_vals_val[jdx])

            X_trains.append(np.array(X_train))
            Y_trains.append(np.array(Y_train))
            X_vals.append(np.array(X_val))
            Y_vals.append(np.array(Y_val))

    for subject_index in new_sub_indices:
        if subject_index not in exclude:
            tree = ET.parse(dir + '/OhioT1DM-2-training/' + str(subject_index) + '-ws-training.xml')
            root = tree.getroot()
            glucose_level_els = [e for e in root.find('glucose_level').findall('event')]
            glucose_level_time_dict = dict(
                [(datetime.datetime.strptime(el.attrib['ts'], '%d-%m-%Y %H:%M:%S'), float(el.attrib['value'])) for el in
                 glucose_level_els])

            # fill in missing values
            times = list(glucose_level_time_dict.keys())

            glucose_level_dict_include_missing = {}

            for idx, t in enumerate(times):
                if idx > 0:
                    diff = (times[idx] - times[idx - 1]).seconds
                    if diff > FIVE_MIN_SECONDS + 10:
                        # add missing values
                        for jdx in range(int((diff / FIVE_MIN_SECONDS) - 1)):
                            temp = times[idx - 1] + datetime.timedelta(seconds=(jdx + 1) * FIVE_MIN_SECONDS)
                            glucose_level_dict_include_missing[temp] = 'MISSING'

                    # add current value
                    glucose_level_dict_include_missing[t] = glucose_level_time_dict[t]
                else:
                    glucose_level_dict_include_missing[t] = glucose_level_time_dict[t]

            glucose_times = list(glucose_level_dict_include_missing.values())

            nb_train_ex = int(0.8 * len(glucose_times))

            glucose_vals_train = glucose_times[:nb_train_ex]
            all_vals_train += [x for x in glucose_vals_train if x != 'MISSING']

            # validation examples should start an hour after start of validation set
            glucose_vals_val = glucose_times[nb_train_ex + NB_TEST_STEPS_TO_SKIP:]

            X_train, Y_train = [], []

            for idx in range(len(glucose_vals_train) - (history_length + prediction_horizon)):
                # discard examples where Y = MISSING, or all X values = MISSING
                if (glucose_vals_train[idx + history_length + prediction_horizon - 1] != MISSING):
                    if include_missing:
                        if glucose_vals_train[idx:idx + history_length].count(MISSING) != len(glucose_vals_train[idx:idx + history_length]):
                            X_train.append(glucose_vals_train[idx:idx + history_length])
                            Y_train.append(glucose_vals_train[idx + history_length + prediction_horizon - 1])
                    else:
                        if MISSING not in glucose_vals_train[idx:idx + history_length]:
                            X_train.append(glucose_vals_train[idx:idx + history_length])
                            Y_train.append(glucose_vals_train[idx + history_length + prediction_horizon - 1])

            X_val, Y_val = [], []

            for idx in range(len(glucose_vals_val) - (history_length + prediction_horizon)):
                # discard examples where Y = MISSING, or all X values = MISSING
                if (glucose_vals_val[idx + history_length + prediction_horizon - 1] != MISSING):
                    if include_missing:
                        if glucose_vals_val[idx:idx + history_length].count(MISSING) != len(glucose_vals_val[idx:idx + history_length]):
                            X_val.append(glucose_vals_val[idx:idx + history_length])
                            Y_val.append(glucose_vals_val[idx + history_length + prediction_horizon - 1])
                    else:
                        if MISSING not in glucose_vals_val[idx:idx + history_length]:
                            X_val.append(glucose_vals_val[idx:idx + history_length])
                            Y_val.append(glucose_vals_val[idx + history_length + prediction_horizon - 1])

            # add first examples in validation set
            nb_additional_ex = history_length + prediction_horizon - 1
            for jdx in range(nb_additional_ex):
                # discard examples where Y = MISSING, or all X values = MISSING
                if glucose_vals_val[jdx] != MISSING:
                    temp = glucose_vals_train + glucose_vals_val
                    temp_sidx = len(glucose_vals_train) - (history_length + prediction_horizon - 1) + jdx

                    if include_missing:
                        if temp[temp_sidx:temp_sidx + history_length].count(MISSING) != len(temp[temp_sidx:temp_sidx + history_length]):
                            X_val.append(temp[temp_sidx:temp_sidx + history_length])
                            Y_val.append(glucose_vals_val[jdx])
                    else:
                        if MISSING not in temp[temp_sidx:temp_sidx + history_length]:
                            X_val.append(temp[temp_sidx:temp_sidx + history_length])
                            Y_val.append(glucose_vals_val[jdx])

            X_trains.append(np.array(X_train))
            Y_trains.append(np.array(Y_train))
            X_vals.append(np.array(X_val))
            Y_vals.append(np.array(Y_val))

    # replace missing values with mean
    mean = np.mean(all_vals_train)
    stdev = np.std(all_vals_train)

    X_trains_missing_replaced, X_vals_missing_replaced = [], []

    # standardize data, replacing missing values with 0
    for train_examples in X_trains:
        X_trains_missing_replaced.append(
            np.array([[(float(x_) - mean) / stdev if x_ != MISSING else 0.0 for x_ in x] for x in train_examples]))
    for val_examples in X_vals:
        X_vals_missing_replaced.append(
            np.array([[(float(x_) - mean) / stdev if x_ != MISSING else 0.0 for x_ in x] for x in val_examples]))

    X_trains = np.array(X_trains_missing_replaced)
    X_vals = np.array(X_vals_missing_replaced)

    X_train = np.concatenate(X_trains)
    Y_train = np.concatenate(Y_trains)
    X_val = np.concatenate(X_vals)
    Y_val = np.concatenate(Y_vals)

    Y_train = np.array([(y - mean) / stdev for y in Y_train])
    Y_val = np.array([(y - mean) / stdev for y in Y_val])

    return X_train, Y_train, X_val, Y_val, mean, stdev

def load_test_data(dir, subject_index, history_length, prediction_horizon, data_mean, data_stdev):
    # load training examples
    tree = ET.parse(dir + '/OhioT1DM-2-training/' + str(subject_index) + '-ws-training.xml')
    root = tree.getroot()
    glucose_level_els = [e for e in root.find('glucose_level').findall('event')]
    glucose_level_time_dict_train = dict(
        [(datetime.datetime.strptime(el.attrib['ts'], '%d-%m-%Y %H:%M:%S'), float(el.attrib['value'])) for el in
         glucose_level_els])

    # fill in missing values
    times = list(glucose_level_time_dict_train.keys())

    glucose_level_dict_include_missing_train = {}

    for idx, t in enumerate(times):
        if idx > 0:
            diff = (times[idx] - times[idx - 1]).seconds
            if diff > FIVE_MIN_SECONDS + 10:
                # add missing values
                for jdx in range(int((diff / FIVE_MIN_SECONDS) - 1)):
                    temp = times[idx - 1] + datetime.timedelta(seconds=(jdx + 1) * FIVE_MIN_SECONDS)
                    glucose_level_dict_include_missing_train[temp] = 'MISSING'

            # add current value
            glucose_level_dict_include_missing_train[t] = glucose_level_time_dict_train[t]
        else:
            glucose_level_dict_include_missing_train[t] = glucose_level_time_dict_train[t]

    glucose_vals_train = list(glucose_level_dict_include_missing_train.values())

    # load test examples
    tree = ET.parse(dir + '/OhioT1DM-2-testing/' + str(subject_index) + '-ws-testing.xml')
    root = tree.getroot()
    glucose_level_els = [e for e in root.find('glucose_level').findall('event')]
    glucose_level_time_dict = dict(
        [(datetime.datetime.strptime(el.attrib['ts'], '%d-%m-%Y %H:%M:%S'), float(el.attrib['value'])) for el in
         glucose_level_els])

    # fill in missing values
    times = list(glucose_level_time_dict.keys())
    test_times_report_only = times[NB_TEST_STEPS_TO_SKIP:]

    glucose_level_dict_include_missing = {}

    for idx, t in enumerate(times):
        if idx > 0:
            diff = (times[idx] - times[idx - 1]).seconds
            if diff > FIVE_MIN_SECONDS + 10:
                # add missing values
                for jdx in range(int((diff / FIVE_MIN_SECONDS) - 1)):
                    temp = times[idx - 1] + datetime.timedelta(seconds=(jdx + 1) * FIVE_MIN_SECONDS)
                    glucose_level_dict_include_missing[temp] = 'MISSING'

            # add current value
            glucose_level_dict_include_missing[t] = glucose_level_time_dict[t]
        else:
            glucose_level_dict_include_missing[t] = glucose_level_time_dict[t]

    glucose_times_test = list(glucose_level_dict_include_missing.values())

    # validation examples should start an hour after start of validation set
    glucose_vals_test = glucose_times_test

    X_test, Y_test = [], []

    for idx in range(1, len(glucose_vals_test) - (history_length + prediction_horizon) + 1):
        if (glucose_vals_test[idx + history_length + prediction_horizon - 1] != MISSING):
            X_test.append(glucose_vals_test[idx:idx + history_length])
            Y_test.append(glucose_vals_test[idx + history_length + prediction_horizon - 1])

    # add first examples in validation set
    nb_additional_ex = history_length + prediction_horizon - NB_TEST_STEPS_TO_SKIP

    X_additional, Y_additional = [], []

    for jdx in range(nb_additional_ex):
        # don't choose examples where Y = MISSING, or all X values = MISSING
        if glucose_vals_test[jdx+NB_TEST_STEPS_TO_SKIP] != MISSING:
            temp = glucose_vals_train + glucose_vals_test
            temp_sidx = len(glucose_vals_train) - (abs(prediction_horizon - history_length)) + jdx + 1
            X_additional.append(temp[temp_sidx:temp_sidx + history_length])
            Y_additional.append(glucose_vals_test[jdx + NB_TEST_STEPS_TO_SKIP])

    X_test = np.array(X_additional + X_test)
    Y_test = np.array(Y_additional + Y_test)

    # standardize data, replacing missing values with 0
    # replace missing values
    X_test_missing_replaced = []

    for val_examples in X_test:
        X_test_missing_replaced.append(
            np.array([(float(x_) - data_mean) / data_stdev if x_ != 'MISSING' else 0.0 for x_ in val_examples]))
    X_test = np.array(X_test_missing_replaced)
    Y_test = np.array([(y - data_mean) / data_stdev for y in Y_test])

    return X_test, Y_test, test_times_report_only