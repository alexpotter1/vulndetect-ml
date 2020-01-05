import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

print(tf.__version__)

### Global vars
BATCH_SIZE = 256
BUFFER_SIZE = 60000
EPOCHES = 300
OUTPUT_DIR = "img"

class Generator(keras.Model):
    def __init__(self, random_noise_size=100):
        super().__init__(name='generator')

        self.input_layer = keras.layers.Dense(units=random_noise_size)
        self.dense_1 = keras.layers.Dense(units=128)
        self.leaky_1 = keras.layers.LeakyReLU(alpha=0.01)
        self.dense_2 = keras.layers.Dense(units=128)
        self.leaky_2 = keras.layers.LeakyReLU(alpha=0.01)
        self.dense_3 = keras.layers.Dense(units=128)
        self.leaky_3 = keras.layers.LeakyReLU(alpha=0.01)
        self.output_layer = keras.layers.Dense(units=784, activation='tanh')

    def call(self, input_tensor):
        x = self.input_layer(input_tensor)
        x = self.dense_1(x)
        x = self.leaky_1(x)
        x = self.dense_2(x)
        x = self.leaky_2(x)
        x = self.dense_3(x)
        x = self.leaky_3(x)
        
        return self.output_layer(x)
    
    def gen_noise(self, batch_size, random_noise_size):
        return np.random.uniform(-1, 1, size=(batch_size, random_noise_size))

class Discriminator(keras.Model):
    def __init__(self):
        super().__init__(name='discriminator')

        self.input_layer = keras.layers.Dense(units = 784)
        self.dense_1 = keras.layers.Dense(units = 128)
        self.leaky_1 =  keras.layers.LeakyReLU(alpha = 0.01)
        self.dense_2 = keras.layers.Dense(units = 128)
        self.leaky_2 = keras.layers.LeakyReLU(alpha = 0.01)
        self.dense_3 = keras.layers.Dense(units = 128)
        self.leaky_3 = keras.layers.LeakyReLU(alpha = 0.01)

        # this neuron indicates if input is real/fake
        self.logits = keras.layers.Dense(units=1)

    def call(self, input_tensor):
        x = self.input_layer(input_tensor)
        x = self.dense_1(x)
        x = self.leaky_1(x)
        x = self.leaky_2(x)
        x = self.leaky_3(x)
        x = self.logits(x)

        return x
    
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator = Generator()
discriminator = Discriminator()
generator_optimizer = keras.optimizers.RMSprop()
discriminator_optimizer = keras.optimizers.RMSprop()

def generator_objective(dx_of_gx):
    return cross_entropy(tf.ones_like(dx_of_gx), dx_of_gx)

def discriminator_objective(d_x, g_z, smooth_factor=0.5):
    real_loss = cross_entropy(tf.ones_like(d_x) + smooth_factor, d_x)
    fake_loss = cross_entropy(tf.zeros_like(g_z), g_z)
    total_loss = real_loss + fake_loss

    return total_loss

@tf.function
def training_step(generator: Generator, discriminator: Discriminator, images:np.ndarray, k:int=1, batch_size=32):
    for _ in range(k):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            noise = generator.gen_noise(batch_size, 100)
            g_z = generator(noise)
            d_x_true = discriminator(images)
            d_x_fake = discriminator(g_z) # dx_of_gx

            discriminator_loss = discriminator_objective(d_x_true, d_x_fake)
            gradients_of_discriminator = disc_tape.gradient(discriminator_loss, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

            generator_loss = generator_objective(d_x_fake)
            gradients_of_generator = gen_tape.gradient(generator_loss, generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

def training(dataset, epoches):
    for epoch in range(epoches):
        seed = np.random.uniform(-1, 1, size=(1, 100))
        for batch in dataset:
            training_step(generator, discriminator, batch, batch_size=BATCH_SIZE, k=1)
        
        if (epoch % 2) == 0:
            fake_image = tf.reshape(generator(seed), shape=(28,28))
            print("{}/{} epoches".format(epoch, epoches))
            plt.imsave("{}/{}.png".format(OUTPUT_DIR, epoch), fake_image, cmap='gray')

            seed = fake_image

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape)

plt.imshow(train_images[1], cmap='gray')

train_images = train_images.astype('float32')
train_images = (train_images - 127.5) / 127.5
train_dataset = tf.data.Dataset.from_tensor_slices(train_images.reshape(train_images.shape[0], 784)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

training(train_dataset, EPOCHES)

fake_image = generator(np.random.uniform(-1, 1, size=(1,100)))
plt.imshow(tf.reshape(fake_image, shape=(28, 28)), cmap='gray')
