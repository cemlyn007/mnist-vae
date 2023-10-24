import jax
import jax.numpy as jnp
from flax import linen as nn


class Encoder(nn.Module):
    latent_dims: int

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        x = nn.Dense(980)(x)
        x = nn.relu(x)
        x = nn.Dense(1024)(x)
        x = nn.relu(x)
        x = nn.Dense(1280)(x)
        x = nn.relu(x)
        latent_mean_and_log_variance = nn.Dense(self.latent_dims + self.latent_dims)(x)
        latent_mean, latent_log_variance = jnp.split(
            latent_mean_and_log_variance, 2, axis=1
        )
        return latent_mean, latent_log_variance


class Decoder(nn.Module):
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.Dense(1280)(x)
        x = nn.relu(x)
        x = nn.Dense(1024)(x)
        x = nn.relu(x)
        x = nn.Dense(784)(x)
        return x