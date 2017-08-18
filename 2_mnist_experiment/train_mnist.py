#!/usr/bin/python


""" Train the MNIST autoencoder. """


import keras.optimizers as optimizers

import cv2

import numpy as np

import mnist_video
import networks


# Number of epochs to train for.
num_epochs = 20
# Learning rate.
learning_rate = 0.001

frame_size = 64
seq_length = 10


def compile_network():
  """ Builds and compiles the network. """
  model = networks.mnist_composite(frame_size, seq_length=seq_length)

  model.summary()

  # Set up optimization.
  rmsprop = optimizers.RMSprop(lr=learning_rate,
                               decay=learning_rate / num_epochs)
  model.compile(optimizer=rmsprop, loss='mean_squared_error',
                metrics=['accuracy'])

  return model

def hacky_conditional_predict(model, sequence):
  """ A hacky way of testing condional decoder models without using a custom RNN
  implementation. Essentially, what it does is it builds up the input vector
  1-by-1 using the results from the previous iterations.
  Args:
    model: The model to test.
    sequence: The sequence to test with.
  Returns:
    The predicted output sequence. """
  sequence = np.expand_dims(sequence, axis=0)
  output_sequence = np.zeros((1, seq_length, frame_size, frame_size, 1))

  for i in range(0, output_sequence.shape[1]):
    # Run an iteration.
    predictions = model.predict([sequence, output_sequence])
    # Add the output frame to the input.
    if i < output_sequence.shape[1] - 1:
      output_sequence[:, i + 1] = predictions[:, i]

  # We've built up the entire output.
  return output_sequence

def shift_sequence(sequence):
  """ Shifts an input sequence one to the right, adding zeros in the first
  place.
  Args:
    sequence: The sequence to shift.
  Returns:
    The shifted sequence. """
  batches = sequence.shape[0]
  first_zeros = np.zeros((batches, 1, frame_size, frame_size, 1))
  print first_zeros.shape
  print sequence.shape
  shifted = np.concatenate((first_zeros, sequence), axis=1)
  return shifted[:, :-1]

def mean_squared_error(expected, actual):
  """ Calculates the MSE between two numpy arrays.
  Args:
    expected: The expected outcomes.
    actual: The actual outcomes. """
  diff = expected - actual
  squared = np.square(diff)
  return np.mean(squared)

def train():
  """ Trains the model. """
  model = compile_network()

  train_data = np.empty((1000, 20, frame_size, frame_size, 1))
  test_data = np.empty((100, 20, frame_size, frame_size, 1))

  # Load MNIST data.
  train, test = mnist_video.load_mnist()
  train_x, _ = train
  test_x, _ = test

  # Training loop.
  for i in range(0, num_epochs):
    # Generate a dataset for this epoch.
    print "Generating dataset..."
    mnist_video.generate_videos(train_x, train_data, frame_size=frame_size)
    print "Done!"

    # Since it's an autoencoder, are "labels" are the same as our input data.
    train_down = train_data[:, 0:seq_length]
    train_flip = train_down[:, ::-1]
    # We need a shifted version of the ground-truth data as the input to the
    # conditional decoder.
    #train_shift = shift_sequence(train_flip)
    model.fit(train_down,
              train_flip, epochs=1,
              batch_size=32)

    mnist_video.generate_videos(test_x, test_data, frame_size=frame_size)

    # Test after each epoch.
    test_down = test_data[:, 0:seq_length]
    test_flip = test_down[:, ::-1]
    #test_outputs = hacky_conditional_predict(model, test_flip[0])
    #loss = mean_squared_error(test_flip[0], test_outputs)
    loss, _ = model.evaluate(test_down, test_flip, batch_size=32)
    print "Loss: %f" % (loss)

    predictions = model.predict(test_down, batch_size=32)

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", 600, 600)
    for i in range(0, seq_length):
      cv2.imshow("image", predictions[0][i] + 0.2)
      cv2.waitKey()


def main():
  train()


if __name__ == "__main__":
  main()
