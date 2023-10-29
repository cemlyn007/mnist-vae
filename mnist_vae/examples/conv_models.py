import jax
import jax.numpy as jnp
from flax import linen as nn


class Encoder(nn.Module):
    latent_dims: int

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        x = x.reshape((-1, 1, 28, 28))
        x = nn.Conv(features=256, kernel_size=(2, 2), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=128, kernel_size=(2, 2), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=96, kernel_size=(2, 2), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(2, 2), strides=(1, 1))(x)
        x = nn.relu(x)
        x = x.reshape((-1, 4 * 64))
        latent_mean_and_log_variance = nn.Dense(self.latent_dims + self.latent_dims)(x)
        latent_mean, latent_log_variance = jnp.split(
            latent_mean_and_log_variance, 2, axis=1
        )
        return latent_mean, latent_log_variance


class Decoder(nn.Module):
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = x.reshape((x.shape[0], 1, 1, -1))
        x = nn.ConvTranspose(
            features=256, kernel_size=(2, 2), strides=(1, 1), padding="VALID"
        )(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(
            features=128, kernel_size=(2, 2), strides=(2, 2), padding="VALID"
        )(x)
        x = nn.relu(x)
        x = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
        x = nn.ConvTranspose(
            features=96, kernel_size=(2, 2), strides=(2, 2), padding="VALID"
        )(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(
            features=64, kernel_size=(2, 2), strides=(1, 1), padding="VALID"
        )(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(
            features=64, kernel_size=(2, 2), strides=(2, 2), padding="VALID"
        )(x)
        x = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
        x = nn.relu(x)
        x = nn.ConvTranspose(
            features=1, kernel_size=(2, 2), strides=(1, 1), padding="SAME"
        )(x)
        x = x.reshape((-1, 784))
        return x


if __name__ == "__main__":
    decoder = Decoder()

    state = decoder.init(jax.random.PRNGKey(0), jnp.ones((1, 16)))

    predictions = decoder.apply(state, jnp.ones((32, 16)))

    assert predictions.shape == (32, 784)
