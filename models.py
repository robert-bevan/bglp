from keras.models import Model
from keras.layers import Dense, LSTM, GRU, Lambda, dot, concatenate, Activation, Input

class LinearModel:
    def __init__(self, input_shape=(6,), nb_output_units=1):
        self.input_shape = input_shape
        self.nb_output_units = nb_output_units

    def __repr__(self):
        return 'Linear'

    def build(self):
        i = Input(shape=self.input_shape)
        x = Dense(self.nb_output_units, activation=None)(i)

        return Model(inputs=[i], outputs=[x])

class MLPModel:
    def __init__(self, input_shape=(6,), nb_output_units=1, nb_hidden_units=128, nb_layers=1, hidden_activation='relu'):
        self.input_shape = input_shape
        self.nb_output_units = nb_output_units
        self.nb_hidden_units = nb_hidden_units
        self.nb_layers = nb_layers
        self.hidden_activation = hidden_activation

    def __repr__(self):
        return 'MLP_{0}_units_{1}_layers'.format(self.nb_hidden_units, self.nb_layers)

    def build(self):
        # input
        i = Input(shape=self.input_shape)

        # add first LSTM layer
        x = Dense(self.nb_hidden_units, input_shape=self.input_shape, activation=self.hidden_activation)(i)

        if self.nb_layers > 1:
            for _ in range(self.nb_layers - 1):
                x = Dense(self.nb_hidden_units, input_shape=self.input_shape, activation=self.hidden_activation)(x)

        x = Dense(self.nb_output_units, activation=None)(x)

        return Model(inputs=[i], outputs=[x])

class GRUModel:
    def __init__(self, input_shape=(6, 1), nb_output_units=1, nb_hidden_units=128, nb_layers=1, dropout=0.0, recurrent_dropout=0.0):
        self.input_shape = input_shape
        self.nb_output_units = nb_output_units
        self.nb_hidden_units = nb_hidden_units
        self.nb_layers = nb_layers
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout

    def __repr__(self):
        return 'GRU_{0}_units_{1}_layers_dropout={2}_{3}'.format(self.nb_hidden_units, self.nb_layers, self.dropout, self.recurrent_dropout)

    def build(self):
        # input
        i = Input(shape=self.input_shape)

        # add first LSTM layer
        x = GRU(self.nb_hidden_units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout, return_sequences=self.nb_layers > 1)(i)

        if self.nb_layers > 1:
            for _ in range(self.nb_layers - 2):
                x = GRU(self.nb_hidden_units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout, return_sequences=True)(x)

            # add final GRU layer
            x = GRU(self.nb_hidden_units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout, return_sequences=False)(x)

        x = Dense(self.nb_output_units, activation=None)(x)

        return Model(inputs=[i], outputs=[x])

class LSTMModel:
    def __init__(self, input_shape=(6, 1), nb_output_units=1, nb_hidden_units=128, nb_layers=1, dropout=0.0, recurrent_dropout=0.0):
        self.input_shape = input_shape
        self.nb_output_units = nb_output_units
        self.nb_hidden_units = nb_hidden_units
        self.nb_layers = nb_layers
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout

    def __repr__(self):
        return 'LSTM_{0}_units_{1}_layers_dropout={2}_{3}'.format(self.nb_hidden_units, self.nb_layers, self.dropout, self.recurrent_dropout)

    def build(self):
        # input
        i = Input(shape=self.input_shape)

        # add first LSTM layer
        x = LSTM(self.nb_hidden_units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout, return_sequences=self.nb_layers > 1)(i)

        if self.nb_layers > 1:
            for _ in range(self.nb_layers - 2):
                x = LSTM(self.nb_hidden_units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout, return_sequences=True)(x)

            # add final LSTM layer
            x = LSTM(self.nb_hidden_units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout, return_sequences=False)(x)

        x = Dense(self.nb_output_units, activation=None)(x)

        return Model(inputs=[i], outputs=[x])

'''
Attention mechanism code based on: https://github.com/philipperemy/keras-attention-mechanism
(Apache License 2.0)
'''
class LSTMAttentionModel:
    def __init__(self, input_shape=(6, 1), nb_output_units=1, nb_hidden_units=128, dropout=0.0, recurrent_dropout=0.0,
                 nb_attention_units=64):
        self.input_shape = input_shape
        self.nb_output_units = nb_output_units
        self.nb_hidden_units = nb_hidden_units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.nb_attention_units = nb_attention_units

    def __repr__(self):
        return 'LSTMAttention_{0}_units_dropout={1}_{2}_{3}_attention_units'.format(self.nb_hidden_units, self.dropout, self.recurrent_dropout, self.nb_attention_units)

    def build(self):
        # input
        i = Input(shape=self.input_shape)

        # LSTM layer
        x = LSTM(self.nb_hidden_units, return_sequences=True)(i)

        # attention mechanism
        score_first_part = Dense(int(x.shape[2]), use_bias=False)(x)
        hidden_state = Lambda(lambda x: x[:, -1, :], output_shape=(self.nb_hidden_units,))(x)
        score = dot([score_first_part, hidden_state], [2, 1])
        attention_weights = Activation('softmax')(score)
        context_vector = dot([x, attention_weights], [1, 1])
        pre_activation = concatenate([context_vector, hidden_state])
        x = Dense(self.nb_attention_units, use_bias=False, activation='tanh')(pre_activation)

        # output
        x = Dense(1, activation=None)(x)

        return Model(inputs=[i], outputs=[x])
