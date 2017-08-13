from keras.models import Model
import keras.backend as K
import keras.layers as layers

from custom_layers import HiddenStateLstm


def mnist_composite():
  """ An LSTM autoencoder that takes in videos of moving MNIST digits and tries
  to reproduce them as well as predict future frames.
  Returns:
    The network, not yet compiled. """
  enc_in = layers.Input(shape=(10, 32, 32, 1))
  # We're going to need to flatten each frame before it can be used as input.
  enc_flat = layers.Reshape((10, 32 * 32 * 1))(enc_in)
  # Encoder LSTM.
  output = layers.LSTM(1024, return_state=True, unroll=True)(enc_flat)
  enc_out = output[0]
  enc_hidden = output[1:]

  # The decoder takes no input, so we're just going to feed it zeros.
  #recon_in = layers.Input(shape=(10, 32, 32, 1))
  #recon_flat = layers.Reshape((10, 32 * 32 * 1))(recon_in)
  enc_seq = layers.RepeatVector(10)(enc_out)
  recon_dec = layers.LSTM(1024, unroll=True,
                          return_sequences=True)(enc_seq,
                                                 initial_state=enc_hidden)

  # Convert to output images.
  recon_out = layers.Reshape((10, 32, 32, 1))(recon_dec)

  # Build model.
  model = Model(inputs=enc_in, outputs=recon_out)
  return model
