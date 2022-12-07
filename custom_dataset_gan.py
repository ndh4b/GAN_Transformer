#!/usr/bin/env python3

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt

image_size = (128, 128)
batch_size = 16

data = tf.keras.preprocessing.image_dataset_from_directory(
    "/home/ndh4b/Project/michelangelo/cropped/",
    image_size=image_size,
    batch_size=batch_size,
    shuffle=True,
    labels=None,
)

real_images = (np.vstack([i.numpy() for i in data.take(1)]))/255

img_rows = 128
img_cols = 128
channels = 3
img_shape = (img_rows, img_cols, channels)
num_classes = 1
latent_dim = 128

# Generator

noise = keras.layers.Input(shape=(latent_dim,))
generator_hidden = keras.layers.Dense(128 * 32 * 32, activation='gelu')(noise)
generator_hidden = keras.layers.Reshape((32, 32, 128))(generator_hidden)
generator_hidden = keras.layers.BatchNormalization(momentum=0.8)(generator_hidden)
generator_hidden = keras.layers.Conv2DTranspose(64, kernel_size=2, strides=2,
                                                activation='gelu')(generator_hidden)
generator_hidden = keras.layers.BatchNormalization(momentum=0.8)(generator_hidden)
generator_hidden = keras.layers.Conv2DTranspose(32, kernel_size=2, strides=2,
                                                activation='gelu')(generator_hidden)
generator_hidden = keras.layers.BatchNormalization(momentum=0.8)(generator_hidden)
g_image = keras.layers.Conv2DTranspose(channels, kernel_size=3,
                                       padding='same', activation='tanh')(generator_hidden)
generator = keras.Model(noise, g_image)
generator.compile(loss=keras.losses.BinaryCrossentropy(),
                  optimizer=keras.optimizers.Adam(0.0001,0.9))

generator.load_weights("michelangelo_gan_weights.h5")

# Discriminator

d_image = keras.layers.Input(shape=img_shape)
discriminator_hidden = keras.layers.Conv2D(16, kernel_size=3, strides=2,
                                           padding='same')(d_image)
discriminator_hidden = keras.layers.LeakyReLU(alpha=0.2)(discriminator_hidden)
discriminator_hidden = keras.layers.Dropout(0.25)(discriminator_hidden)
discriminator_hidden = keras.layers.Conv2D(32, kernel_size=3, strides=2,
                                           padding='same')(discriminator_hidden)
discriminator_hidden = keras.layers.ZeroPadding2D(padding=((0,1),(0,1)))(discriminator_hidden)
discriminator_hidden = keras.layers.LeakyReLU(alpha=0.2)(discriminator_hidden)
discriminator_hidden = keras.layers.Dropout(0.25)(discriminator_hidden)
discriminator_hidden = keras.layers.BatchNormalization(momentum=0.8)(discriminator_hidden)
discriminator_hidden = keras.layers.Conv2D(64, kernel_size=3, strides=2,
                                           padding='same')(discriminator_hidden)
discriminator_hidden = keras.layers.LeakyReLU(alpha=0.2)(discriminator_hidden)
discriminator_hidden = keras.layers.Dropout(0.25)(discriminator_hidden)
discriminator_hidden = keras.layers.BatchNormalization(momentum=0.8)(discriminator_hidden)
discriminator_hidden = keras.layers.Conv2D(128, kernel_size=3, strides=1,
                                           padding='same')(discriminator_hidden)
discriminator_hidden = keras.layers.LeakyReLU(alpha=0.2)(discriminator_hidden)
discriminator_hidden = keras.layers.Dropout(0.25)(discriminator_hidden)
discriminator_hidden = keras.layers.Flatten()(discriminator_hidden)

valid = keras.layers.Dense(1, activation='sigmoid')(discriminator_hidden)
discriminator = keras.Model(d_image, valid)
discriminator.compile(loss=[keras.losses.BinaryCrossentropy(),
                            keras.losses.SparseCategoricalCrossentropy()],
                      optimizer=keras.optimizers.Adam(0.00002,0.5),
                      metrics=['accuracy'])

# Combined model

discriminator.trainable = False
valid = discriminator(g_image)
combined = keras.Model(noise,valid)
combined.compile(loss=[keras.losses.BinaryCrossentropy(),
                       keras.losses.SparseCategoricalCrossentropy()],
                 optimizer=keras.optimizers.Adam(0.0002,0.5))

history = [[],[],[]]
half_batch_size = int(batch_size/2)
batches = 15000

for batch in range(batches):    
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    generated_images = generator.predict(noise)
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1),dtype=float)
    d_loss_real = discriminator.train_on_batch(real_images, valid)
    d_loss_fake = discriminator.train_on_batch(generated_images, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
     #  Training the generator...    
    valid = 0.5 * np.ones((batch_size, 1))
    # Train the generator
    g_loss = combined.train_on_batch(noise, valid)
    history[0] += [d_loss[0]]
    history[1] += [d_loss[1]]
    history[2] += [g_loss]
    
    print("\r%d" % (batch))
    print("Disc. Loss: %f" % (d_loss[0]))
    print("Real/Fake-Acc.: %.2f%%" % (100*d_loss[1]))
    print("Gen. Loss: %f" % (g_loss))
    
generator.save_weights("michelangelo_gan_weights.h5",save_format='h5')