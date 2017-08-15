""" Utility for generating a simple dataset of moving MNIST images on-the-fly.
"""


import cPickle as pickle
import gzip
import random

import numpy as np


def load_mnist():
  """ Loads the MNIST dataset.
  Returns: Two tuples. The first contains the training data and labels, and the
           second contains the testing data and labels. """
  mnist_file = gzip.open("mnist.pkl.gz", "rb")
  train_set, test_set, _ = pickle.load(mnist_file)
  mnist_file.close()

  train_x, train_y = train_set
  test_x, test_y = test_set

  train_x = train_x.reshape(-1, 28, 28, 1)
  test_x = test_x.reshape(-1, 28, 28, 1)

  return ((train_x, train_y), (test_x, test_y))

def _draw_number(number, pos, frame):
  """ Draws a single number in the frame.
  Args:
    number: The number image.
    pos: The position to draw at.
    frame: The frame to draw in. """
  width, height, _ = number.shape

  # Calculate box.
  num_x, num_y = pos
  num_x_start = num_x - width / 2
  num_x_end = num_x + width / 2
  num_y_start = num_y - width / 2
  num_y_end = num_y + width / 2

  # We make this additive to two numbers can safely overlap.
  frame[num_y_start:num_y_end, num_x_start:num_x_end] += number
  frame.clip(0, 1)

  return frame

def _draw_frame(number_1, number_2, num_1_pos, num_2_pos, frame):
  """ Draw a single frame.
  Args:
    number_1: The image of the first number.
    number_2: The image of the second number.
    num_1_pos: The position to draw the first number at (center)
    num_2_pos: The position to draw the second number at (center).
    frame: The frame to draw in. """
  # Draw the numbers.
  _draw_number(number_1, num_1_pos, frame)
  #_draw_number(number_2, num_2_pos, frame)

def _make_video(number_1, number_2, video, frame_size):
  """ Make a 20-frame video of two MNIST numbers moving.
  Args:
    number_1: The image of the first number.
    number_2: The image of the second number.
    video: The array to write the output video to. Should contain zeros.
    frame_size: The size of each square frame. """
  width, height, _ = number_1.shape

  # To begin with, choose random positions for the numbers. We want to make sure
  # that they're not right up against the walls.
  num_1_x = random.randint(width / 2 + 1, frame_size - width / 2 - 1)
  num_2_x = random.randint(width / 2 + 1, frame_size - width / 2 - 1)
  num_1_y = random.randint(height / 2 + 1, frame_size - height / 2 - 1)
  num_2_y = random.randint(height / 2 + 1, frame_size - height / 2 - 1)

  # Choose random velocity vectors.
  num_1_vel = np.array([random.random() - 0.5, random.random() - 0.5])
  # The speed should always be one.
  num_1_vel /= np.linalg.norm(num_1_vel)

  num_2_vel = np.array([random.random() - 0.5, random.random() - 0.5])
  num_2_vel /= np.linalg.norm(num_2_vel)

  num_1_pos = np.array([num_1_x, num_1_y], dtype="float")
  num_2_pos = np.array([num_2_x, num_2_y], dtype="float")

  # Compute 20 frames.
  for i in range(0, 20):
    # Draw the frame.
    _draw_frame(number_1, number_2, num_1_pos.astype("int"),
                num_2_pos.astype("int"), video[i])

    # Update the number positions.
    num_1_pos += num_1_vel
    num_2_pos += num_2_vel

    # Bounce if we hit a wall.
    half_width = width / 2
    half_height = height / 2
    num_1_x, num_1_y = num_1_pos.astype("int")
    num_2_x, num_2_y = num_2_pos.astype("int")

    if (num_1_x >= frame_size - half_width or num_1_x <= half_width):
      num_1_vel[0] *= -1.0
    if (num_1_y >= frame_size - half_height or num_1_y <= half_height):
      num_1_vel[1] *= -1.0
    if (num_2_x >= frame_size - half_width or num_2_x <= half_width):
      num_2_vel[0] *= -1.0
    if (num_2_y >= frame_size - half_height or num_2_y <= half_height):
      num_2_vel[1] *= -1.0

def generate_videos(mnist_set, batch, frame_size=64):
  """ Generates a batch of random videos.
  Args:
    mnist_set: The array of MNIST digits to choose from.
    batch: The batch array to write into. Should be of shape
           (batch_size, 20, 64, 64, 1).
    frame_size: The size of each square frame.
  """
  # Zero out the array initially.
  batch.fill(0)

  for i in range(0, len(batch)):
    # Choose two random digits.
    digit_1 = mnist_set[random.randint(0, len(mnist_set) - 1)]
    digit_2 = mnist_set[random.randint(0, len(mnist_set) - 1)]

    # Create and save the video.
    _make_video(digit_1, digit_2, batch[i], frame_size)

  # Scale and center the data.
  mean = np.mean(batch)
  print mean
  batch -= mean
