#!/usr/bin/env python3

import tensorflow.keras as keras
import numpy as np

img_rows = 32
img_cols = 32
channels = 3
img_shape = (img_rows, img_cols, channels)
num_classes = 10
latent_dim = 128

# Generator

noise = keras.layers.Input(shape=(latent_dim,))
label = keras.layers.Input(shape=(1,), dtype='int32')
label_embedding = keras.layers.Flatten()(keras.layers.Embedding(num_classes, latent_dim)(label))
generator_input = keras.layers.Multiply()([noise, label_embedding])
generator_hidden = keras.layers.Dense(128 * 8 * 8, activation='gelu')(generator_input)
generator_hidden = keras.layers.Reshape((8, 8, 128))(generator_hidden)
generator_hidden = keras.layers.BatchNormalization(momentum=0.8)(generator_hidden)
generator_hidden = keras.layers.Conv2DTranspose(64, kernel_size=2, strides=2,
                                                activation='gelu')(generator_hidden)
generator_hidden = keras.layers.BatchNormalization(momentum=0.8)(generator_hidden)
generator_hidden = keras.layers.Conv2DTranspose(32, kernel_size=2, strides=2,
                                                activation='gelu')(generator_hidden)
generator_hidden = keras.layers.BatchNormalization(momentum=0.8)(generator_hidden)
g_image = keras.layers.Conv2DTranspose(channels, kernel_size=3,
                                       padding='same', activation='tanh')(generator_hidden)
generator = keras.Model([noise, label], g_image)
generator.compile(loss=keras.losses.BinaryCrossentropy(),
                  optimizer=keras.optimizers.Adam(0.0001,0.9))

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
target_label = keras.layers.Dense(num_classes+1,
                                  activation='softmax')(discriminator_hidden)
discriminator = keras.Model(d_image, [valid, target_label])
discriminator.compile(loss=[keras.losses.BinaryCrossentropy(),
                            keras.losses.SparseCategoricalCrossentropy()],
                      optimizer=keras.optimizers.Adam(0.00002,0.5),
                      metrics=['accuracy'])

# Combined model

discriminator.trainable = False
valid, target_label = discriminator(g_image)
combined = keras.Model([noise,label],[valid,target_label])
combined.compile(loss=[keras.losses.BinaryCrossentropy(),
                       keras.losses.SparseCategoricalCrossentropy()],
                 optimizer=keras.optimizers.Adam(0.0002,0.5))

(x_train, y_train),(x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train[0:1000]
y_train = y_train[0:1000]
x_test = x_test[0:1000]
y_test = y_test[0:1000]
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = y_train.reshape(-1, 1)

history = [[],[],[],[]]
batch_size = 256
half_batch_size = int(batch_size/2)

batches = 30000

generator.load_weights("cifar10_gan_weights.h5")

for batch in range(batches):
    idx = np.random.randint(0, x_train.shape[0], half_batch_size)
    real_images = x_train[idx]
    noise = np.random.normal(0, 1, (half_batch_size, latent_dim))
    sampled_labels = np.random.randint(0, num_classes, half_batch_size).reshape(-1, 1)
    generated_images = generator.predict([noise, sampled_labels])
    valid = np.ones((half_batch_size, 1))
    fake = np.zeros((half_batch_size, 1),dtype=float)
    image_labels = y_train[idx]
    fake_labels = (num_classes) * np.ones(half_batch_size).reshape(-1, 1)

    d_loss_real = discriminator.train_on_batch(real_images, [valid, image_labels])
    d_loss_fake = discriminator.train_on_batch(generated_images, [fake, fake_labels])
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
    #  Training the generator...

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    valid = 0.5 * np.ones((batch_size, 1))
    sampled_labels = np.random.randint(0, num_classes, batch_size).reshape(-1, 1)

    # Train the generator
    
    g_loss = combined.train_on_batch([noise, sampled_labels], [valid, sampled_labels])

    history[0] += [d_loss[0]]
    history[1] += [d_loss[3]]
    history[2] += [d_loss[4]]
    history[3] += [g_loss[0]]
    
    print("\r%d" % (batch))
    print("Disc. Loss: %f" % (d_loss[0]))
    print("Real/Fake-Acc.: %.2f%%" % (100*d_loss[3]))
    print("Class Acc.: %.2f%%" % (100*d_loss[4]))
    print("Gen. Loss: %f" % (g_loss[0]))
    
generator.save_weights("cifar10_gan_weights.h5",save_format='h5')
