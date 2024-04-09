import tensorflow as tf
import keras

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, Layer
import keras.backend as K

def linear_regression(xtrain,ytrain):
    
    from sklearn.linear_model import LinearRegression

    L = LinearRegression()
    L.fit(xtrain,ytrain)
    ytrain_hat = L.predict(xtrain)
    r_squared = r2_score(ytrain, ytrain_hat, multioutput='variance_weighted')
    
    return r_squared, L


def VAE(xtrain, ytrain, latent_dim=6, beta=.1, num_epochs=100, anneal_step=.0001, pretrained_encoder=[]):
    
    from keras import backend as K
    from keras import layers
    from keras.callbacks import EarlyStopping
    
    from numpy.random import seed
    seed(1)
    keras.utils.set_random_seed(1)

    input_dim = xtrain.shape[1]
    output_dim = ytrain.shape[1]
    latent_dim = latent_dim
    input_data = keras.Input(shape=(input_dim,))
    
    class Sampling(Layer):
        """Uses (z_mean, z_log_var) to sample z"""

        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = K.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * .1*epsilon
        
    encoded = Dense(latent_dim, activation='tanh')(input_data)
        
    z_mean = Dense(latent_dim, name="z_mean")(encoded)
    z_log_var = Dense(latent_dim, name="z_log_var")(encoded)
    z = Sampling()([z_mean, z_log_var])
    latent_inputs = keras.Input(shape=(latent_dim,))
    decoded = Dense(10, activation='tanh')(latent_inputs)
    decoded = Dense(output_dim, activation='linear')(decoded)
    
    encoder = keras.Model(input_data, [z_mean, z_log_var, z], name="encoder")
    decoder = keras.Model(latent_inputs, decoded, name="decoder")
    
    if pretrained_encoder:
        encoder.load_weights(pretrained_encoder)
        for layer in encoder.layers:
            layer.trainable = False
    
    class VAE(keras.Model):
        def __init__(self, encoder, decoder, **kwargs):
            super(VAE, self).__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder
            self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
            self.reconstruction_loss_tracker = keras.metrics.Mean(
                name="reconstruction_loss"
            )
            self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
            self.anneal_step = anneal_step
            self.beta = beta 
            
        @property
        def metrics(self):
            return [
                self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker,
            ]

        def train_step(self, data):
            
            data1, data2 = data
            
            with tf.GradientTape() as tape:
                
                z_mean, z_log_var, z = self.encoder(data1)
                
                reconstruction = self.decoder(z)
                reconstruction_loss = K.mean(keras.losses.mse(data2, reconstruction))
                
                # Calculate KL annealing factor
                kl_anneal = tf.cast(
                    tf.where(
                        self.optimizer.iterations < self.anneal_step,
                        self.beta * (self.optimizer.iterations / self.anneal_step),
                        self.beta
                    ),
                    dtype=tf.float32  # Cast to float32 or your desired data type
                )

                kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                kl_loss = kl_anneal * K.mean(kl_loss)
            
                total_loss = reconstruction_loss + kl_loss
                                
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)

            return {
                "loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
                "kl_anneal": kl_anneal
            }
        
        def predict(self, data):
            z_mean,_,_ = self.encoder(data)
            decoded = self.decoder(z_mean)
            return decoded
                   
    vae = VAE(encoder, decoder)
    opt = keras.optimizers.Adam(learning_rate=.01)
    vae.compile(optimizer=opt)
    history = vae.fit(xtrain, ytrain, epochs=num_epochs, batch_size=1000, verbose=True)
        
    r_squared = r2_score(ytrain, vae.predict(xtrain), multioutput = 'variance_weighted')

    return r_squared, vae, encoder, decoder