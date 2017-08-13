#!/usr/bin/python


""" Train the MNIST autoencoder. """


import keras.optimizers as optimizers

import cv2

import numpy as np

import mnist_video
import networks


# Number of epochs to train for.
num_epochs = 10
# Learning rate.
learning_rate = 0.001


def compile_network():
  """ Builds and compiles the network. """
  model = networks.mnist_composite()

  print model.summary()

  # Set up optimization.
  rmsprop = optimizers.RMSprop(lr=learning_rate,
                               decay=learning_rate / num_epochs)
  model.compile(optimizer=rmsprop, loss='mean_squared_error',
                metrics=['accuracy'])

  return model

def train():
  """ Trains the model. """
  model = compile_network()

  train_data = np.empty((1000, 20, 64, 64, 1))
  test_data = np.empty((100, 20, 64, 64, 1))

  # Load MNIST data.
  train, test = mnist_video.load_mnist()
  train_x, _ = train
  test_x, _ = test

  # Zero array to feed the decoder input.
  train_zeros = np.zeros((1000, 10, 32, 32, 1))
  test_zeros = np.zeros((100, 10, 32, 32, 1))

  # Training loop.
  for i in range(0, num_epochs):
    # Generate a dataset for this epoch.
    print "Generating dataset..."
    mnist_video.generate_videos(train_x, train_data)
    print "Done!"

    # Since it's an autoencoder, are "labels" are the same as our input data.
    train_down = train_data[:, 0:10, ::2, ::2]
    train_flip = train_down[:, ::-1]
    model.fit(train_down,
              train_flip, epochs=1,
              batch_size=10)

    mnist_video.generate_videos(test_x, test_data)

    # Test after each epoch.
    test_down = test_data[:, 0:10, ::2, ::2]
    test_flip = test_down[:, ::-1]
    loss, acc = model.evaluate(test_down,
                               test_flip,
                               batch_size=10)
    print "Loss: %f, accuracy: %f" % (loss, acc)

    predictions = model.predict(test_down,
                                batch_size=10)
    for i in range(0, 10):
      cv2.imshow("test", predictions[0][i])
      cv2.waitKey()


def main():
  train()


if __name__ == "__main__":
  main()
