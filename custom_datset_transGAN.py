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

#Transformer Block

class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = keras.layers.MultiHeadAttention(num_heads=num_heads,
                                                   key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [keras.layers.Dense(ff_dim, activation="gelu"),
             keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class PositionEmbedding(keras.layers.Layer):
    def __init__(self, maxlen, embed_dim):
        super(PositionEmbedding, self).__init__()
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-2] 
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions
    
patch_size = 8
embed_dim = 256
num_heads = 12
ff_dim = 1024
stack = 12

#Discriminator

x = keras.layers.Input(shape=img_shape)
y = x    
y = keras.layers.Conv2D(embed_dim,
                            kernel_size=(patch_size),
                            strides=(patch_size,patch_size))(y)
y = keras.layers.Reshape((-1,embed_dim))(y)
    
c = keras.layers.Lambda(lambda x: tf.tile(tf.constant([[[0.0]]]),
                                             (tf.shape(x)[0],1,embed_dim)))(x)
y = keras.layers.Concatenate(axis=1)([c,y])
y = PositionEmbedding(y.shape[1],embed_dim)(y)   
for _ in range(stack):
    y = TransformerBlock(embed_dim,num_heads,ff_dim)(y)
y = keras.layers.Lambda(lambda x: x[:,0,:])(y)  
y = keras.layers.Dense(ff_dim)(y)
y = keras.layers.LayerNormalization()(y)
y = keras.layers.Activation(activation=keras.activations.gelu)(y) 
valid = keras.layers.Dense(1, activation='sigmoid')(y)
discriminator = keras.Model(x, valid)
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
batches = 5000

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
    
generator.save_weights("michelangelo_transGAN_weights.h5",save_format='h5')
