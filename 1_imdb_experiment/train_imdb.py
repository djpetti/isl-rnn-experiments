#!/usr/bin/python


""" Train the IMDB review classifier. """


import keras.optimizers as optimizers

import numpy as np

import imdb
import networks


# Number of epochs to train for.
num_epochs = 10
# Learning rate.
learning_rate = 0.001


def compile_network():
  """ Builds and compiles the network. """
  model = networks.imdb_network()

  print model.summary()

  # Set up optimization.
  rmsprop = optimizers.RMSprop(lr=learning_rate,
                               decay=learning_rate / num_epochs)
  model.compile(optimizer=rmsprop, loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  return model

def train():
  """ Trains the model. """

  # Load the data.
  train, test, _ = imdb.load_data(n_words=10000, maxlen=101)
  train_x, train_y = train
  test_x, test_y = test

  train_x, _, train_y = imdb.prepare_data(train_x, train_y, maxlen=101)
  # It expects the batch dimension first.
  train_x = np.transpose(train_x, (1, 0))
  test_x, _, test_y = imdb.prepare_data(test_x, test_y, maxlen=101)
  test_x = np.transpose(test_x, (1, 0))

  model = compile_network()

  # Training loop.
  for i in range(0, num_epochs):
    model.fit(train_x, train_y, epochs=1, batch_size=100)

    # Test after each epoch.
    loss, acc = model.evaluate(test_x, test_y, batch_size=100)
    print "Loss: %f, accuracy: %f" % (loss, acc)


def main():
  train()


if __name__ == "__main__":
  main()
