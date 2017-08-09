from keras.models import Model
import keras.layers as layers


def imdb_network():
  """ Basic LSTM network for the IMDB dataset.
  Returns:
    The network, not yet compiled. """
  main_in = layers.Input(shape=(100,), dtype="int32")
  # Embedding layer, to convert inputs to dense vectors.
  vector_in = layers.Embedding(output_dim=256, input_dim=10000,
                               input_length=100)(main_in)

  # LSTM portion.
  lstm = layers.LSTM(64)(vector_in)

  # Decoder portion.
  dense1 = layers.Dense(64, activation='relu')(lstm)
  dense2 = layers.Dense(64, activation='relu')(dense1)
  dense3 = layers.Dense(64, activation='relu')(dense2)

  # Simple binary classifier for positive vs. negative.
  preds = layers.Dense(2, activation='softmax')(dense3)

  # Build model.
  model = Model(inputs=main_in, outputs=preds)
  return model
