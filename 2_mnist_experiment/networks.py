from keras.models import Model
import keras.backend as K
import keras.layers as layers

from custom_layers import HiddenStateLstm


def mnist_composite(frame_size, conditional=False, seq_length=10):
  """ An LSTM autoencoder that takes in videos of moving MNIST digits and tries
  to reproduce them as well as predict future frames.
  Args:
    frame_size: The size of one side of an input frame.
    conditional: Whether to use a conditional decoder.
    seq_length: The length of a sequence.
  Returns:
    The network, not yet compiled. """
  enc_in = layers.Input(shape=(seq_length, frame_size, frame_size, 1))
  # We're going to need to flatten each frame before it can be used as input.
  enc_flat = layers.Reshape((seq_length, frame_size * frame_size * 1))(enc_in)
  # Encoder LSTM.
  lstm1 = layers.LSTM(frame_size * frame_size,
                      return_state=True)(enc_flat)

  enc_hidden = lstm1[1:]
  enc_output = lstm1[0]

  dec_input = None
  if conditional:
    # We feed the decoder the previous prediction. For the training case, these
    # are actually the ground-truth images.
    pred_in = layers.Input(shape=(seq_length, frame_size, frame_size, 1))
    pred_flat = layers.Reshape((seq_length, frame_size * frame_size * 1))(pred_in)
    dec_input = pred_flat
  else:
    dec_input = layers.RepeatVector(seq_length)(enc_output)

  recon_dec = layers.LSTM(frame_size * frame_size,
                          return_sequences=True)(dec_input,
                                                 initial_state=enc_hidden)

  # Convert to output images.
  recon_out = layers.Reshape((seq_length, frame_size, frame_size, 1))(recon_dec)

  # Build model.
  input_list = None
  if conditional:
    input_list = [enc_in, pred_in]
  else:
    input_list = [enc_in]
  model = Model(inputs=input_list, outputs=recon_out)
  return model
