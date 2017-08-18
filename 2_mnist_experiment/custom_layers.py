from keras.layers import Dense, multiply, add, Activation, Input
from keras.models import Model

from recurrentshop import basic_cells


class SizeChangingLSTMCell(basic_cells.LSTMCell):
  """ Special LSTM cell that allows the sizes of the inputs and outputs to be
  different from that of the hidden state. """

  def __init__(self, *args, **kwargs):
    """
    Args:
      state_dim: Allows user to specify the shape of the state. It defaults to
								 the output shape. """
    state_dim = kwargs.get("state_dim", None)
    output_dim = kwargs.get("output_dim", None)
    if not state_dim:
      # If not specified, it is the output shape.
      self.state_dim = output_dim
    else:
      self.state_dim = state_dim

    # Remove the state_dim kwarg, because Keras doesn't understand it.
    if "state_dim" in kwargs:
      kwargs.pop("state_dim")

    super(SizeChangingLSTMCell, self).__init__(*args, **kwargs)

  def build_model(self, input_shape):
		output_dim = self.output_dim
		input_dim = input_shape[-1]
		state_dim = self.state_dim

		output_shape = (input_shape[0], output_dim)
		state_shape = (input_shape[0], state_dim)

		x = Input(batch_shape=input_shape)
		h_tm1 = Input(batch_shape=output_shape)
		c_tm1 = Input(batch_shape=state_shape)

		f = add([Dense(state_dim)(x), Dense(state_dim, use_bias=False)(h_tm1)])
		f = Activation('sigmoid')(f)

		i = add([Dense(state_dim)(x), Dense(state_dim, use_bias=False)(h_tm1)])
		i = Activation('sigmoid')(i)
		c_prime = add([Dense(state_dim)(x), Dense(state_dim, use_bias=False)(h_tm1)])
		c_prime = Activation('tanh')(c_prime)

		c = add([multiply([f, c_tm1]), multiply([i, c_prime])])
		c = Activation('tanh')(c)

		#o = add([Dense(output_dim)(x), Dense(output_dim, use_bias=False)(h_tm1)])
		#h = multiply([o, c])
		h = Dense(output_dim)(c)

		return Model([x, h_tm1, c_tm1], [h, h, c])
