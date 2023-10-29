from flax import linen as nn
import jax
import typing
from typing import Callable
from mnist_vae._src import mnist
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import functools
import operator
import sys


class Encoder(nn.Module):
    latent_dims: int

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Accepts a batched flattened 28x28 image.
        Returns a tuple of (latent_mean, latent_log_variance)"""
        raise NotImplementedError("Encoder not implemented!")


class Decoder(nn.Module):
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Accepts a batched latent vector.
        Returns a batched flattened 28x28 image."""
        raise NotImplementedError("Decoder not implemented!")


class Hyperparameters(typing.NamedTuple):
    latent_dims: int
    learning_rate: float


class ModelVariables(typing.NamedTuple):
    encoder: dict[str, any]
    decoder: dict[str, any]


class State(typing.NamedTuple):
    variables: ModelVariables
    optimizer_state: optax.OptState
    key: jax.random.PRNGKey
    step: jax.Array


class Experiment:
    def __init__(
        self,
        encoder_factory: Callable[[int], Encoder],
        decoder_factory: Callable[[], Decoder],
        hyperparameters: Hyperparameters,
        checkpoint_directory: str,
        cache_directory: str,
        backend: str | None = None,
    ) -> None:
        self._device = jax.devices(backend)[0]
        self._checkpoint_directory = checkpoint_directory
        self._checkpoint_manager = self._get_checkpoint_manager(
            self._checkpoint_directory, 0, 0
        )
        self.train_images, self.train_labels = self._get_training_data(cache_directory)
        self._encoder = encoder_factory(hyperparameters.latent_dims)
        self._decoder = decoder_factory()
        self._optimizer = self._get_optimizer(hyperparameters.learning_rate)
        self._last_batch_size = sys.maxsize
        self._state = self._initial_state(jax.random.PRNGKey(0))

        self._train_step = jax.jit(
            self._train_step, static_argnames=("batch_size",), device=self._device
        )
        self._predict = jax.jit(self._predict, device=self._device)
        self._encode = jax.jit(self._encode, device=self._device)

    def reset(self, key: int) -> None:
        if self._checkpoint_manager.directory.exists():
            self._checkpoint_manager.directory.rmtree()
        self._checkpoint_manager.directory.mkdir()
        # else...
        self._state = self._initial_state(jax.random.PRNGKey(key))

    def restore(self, step: int | None = None) -> None:
        state = self._initial_state(jax.random.PRNGKey(0))
        if step is None:
            step = self._checkpoint_manager.latest_step()

        if step is not None:
            self._state = self._checkpoint_manager.restore(
                step,
                items=state,
            )
        else:
            raise RuntimeError("No checkpoint found!")

    def close(self) -> None:
        try:
            step = int(self._state.step.item())
            if self._checkpoint_manager.latest_step() != step:
                self._checkpoint_manager.save(
                    self._state.step.item(), self._state, force=True
                )
        finally:
            self._checkpoint_manager.close()

    def checkpoint_exists(self) -> bool:
        return self._checkpoint_manager.latest_step() is not None

    def _get_checkpoint_manager(
        self,
        checkpoint_directory: str,
        save_interval_steps: int,
        max_to_keep: int,
    ) -> ocp.CheckpointManager:
        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep if max_to_keep else None,
            save_interval_steps=save_interval_steps,
        )
        checkpoint_manager = ocp.CheckpointManager(
            checkpoint_directory, ocp.PyTreeCheckpointer(), options=options
        )
        self._checkpoint_max_to_keep = max_to_keep
        self._checkpoint_interval = save_interval_steps
        return checkpoint_manager

    @property
    def checkpoint_max_to_keep(self) -> int:
        return self._checkpoint_max_to_keep

    @property
    def checkpoint_interval(self) -> int:
        return self._checkpoint_interval

    def update_checkpoint_manager(
        self, save_interval_steps: int, max_to_keep: int
    ) -> None:
        self._checkpoint_manager.close()
        self._checkpoint_manager = self._get_checkpoint_manager(
            self._checkpoint_directory, max_to_keep, save_interval_steps
        )

    def _get_training_data(self, cache_directory: str) -> tuple[jax.Array, jax.Array]:
        dataset = mnist.Dataset(cache_directory)
        dataset.download()
        train_images = dataset.load_train_images(self._device)
        train_images = train_images / 255.0
        train_images = train_images.reshape(
            (
                train_images.shape[0],
                functools.reduce(operator.mul, train_images.shape[1:], 1),
            )
        )
        train_labels = dataset.load_train_labels(self._device)
        return train_images, train_labels

    def _get_optimizer(self, learning_rate: float) -> optax.GradientTransformation:
        return optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate)

    def _initial_state(
        self,
        key: jax.random.PRNGKey,
    ) -> State:
        key, encoder_init_key, decoder_init_key = jax.random.split(key, 3)
        encoder_variables = self._encoder.init(
            encoder_init_key,
            jnp.empty(
                (
                    1,
                    28 * 28,
                ),
                dtype=jnp.float32,
            ),
        )
        decoder_variables = self._decoder.init(
            decoder_init_key,
            jnp.empty(
                (
                    1,
                    self._encoder.latent_dims,
                ),
                dtype=jnp.float32,
            ),
        )
        variables = ModelVariables(encoder=encoder_variables, decoder=decoder_variables)
        optimizer_state = self._optimizer.init(variables)
        return State(
            variables=variables,
            optimizer_state=optimizer_state,
            key=key,
            step=jnp.array(0, dtype=jnp.uint32),
        )

    def train_step(
        self, learning_rate: float, beta: float, batch_size: int
    ) -> dict[str, any]:
        learning_rate = jnp.float32(learning_rate)
        if learning_rate != self._state.optimizer_state.hyperparams["learning_rate"]:
            self._optimizer = self._get_optimizer(learning_rate)
            self._state = self._state._replace(
                optimizer_state=self._optimizer.init(self._state.variables)
            )
        if batch_size != self._last_batch_size:
            self._train_step._clear_cache()
            self._last_batch_size = batch_size
        self._state, metrics = self._train_step(self._state, beta, batch_size)
        step = self._state.step.item()
        if self.checkpoint_interval and (
            self.checkpoint_max_to_keep is None or self.checkpoint_max_to_keep
        ):
            self._checkpoint_manager.save(step, self._state, metrics=metrics)
        metrics = {"step": step, **jax.tree_map(lambda x: x.item(), metrics)}
        return metrics

    def _train_step(
        self, state: State, beta: float, batch_size: int
    ) -> tuple[State, dict[str, any]]:
        new_key, key = jax.random.split(state.key)
        (loss, extras), grad = jax.value_and_grad(self._get_loss, has_aux=True)(
            state.variables, key, beta, batch_size
        )
        updates, new_optimizer_state = self._optimizer.update(
            grad, state.optimizer_state
        )
        new_variables = optax.apply_updates(state.variables, updates)
        next_state = state._replace(
            variables=new_variables,
            optimizer_state=new_optimizer_state,
            key=new_key,
            step=state.step + 1,
        )
        return next_state, {
            "loss": loss,
            "beta": beta,
            "batch_size": batch_size,
            "learning_rate": new_optimizer_state.hyperparams["learning_rate"],
            **extras,
        }

    def _get_loss(
        self,
        model_variables: ModelVariables,
        key: jax.random.PRNGKey,
        beta: float,
        batch_size: int,
    ) -> tuple[jax.Array, dict[str, any]]:
        sample_key, key = jax.random.split(key)
        train_images = self.train_images[
            jax.random.randint(
                sample_key, (batch_size,), 0, self.train_images.shape[0] + 1
            )
        ]
        latent_mean, latent_log_variance = self._encoder.apply(
            model_variables.encoder, train_images
        )
        if latent_mean.shape != (batch_size, self._encoder.latent_dims):
            raise ValueError(
                f"Latent mean shape {latent_mean.shape} does not match the expected shape {(batch_size,)}"
            )
        elif latent_log_variance.shape != (batch_size, self._encoder.latent_dims):
            raise ValueError(
                f"Latent log variance shape {latent_log_variance.shape} does not match the expected shape {(batch_size,)}"
            )
        # else...
        sampled_latent = latent_mean + jnp.exp(
            latent_log_variance * 0.5
        ) * jax.random.normal(key, latent_log_variance.shape)
        predictions = self._decoder.apply(model_variables.decoder, sampled_latent)

        if predictions.shape != train_images.shape:
            raise ValueError(
                f"Predictions shape {predictions.shape} does not match train images shape {train_images.shape}"
            )
        # else...

        binary_cross_entropy = optax.sigmoid_binary_cross_entropy(
            predictions, train_images
        )
        mean_binary_cross_entropy = jnp.mean(binary_cross_entropy)
        kl_divergence = -0.5 * jnp.mean(
            1.0
            + latent_log_variance
            - jnp.square(latent_mean)
            - jnp.exp(latent_log_variance)
        )
        loss = mean_binary_cross_entropy + beta * kl_divergence
        return loss, {
            "mean_binary_cross_entropy": mean_binary_cross_entropy,
            "kl_divergence": kl_divergence,
        }

    def encode_decode(self, key: int, image_indices: list[int]) -> jax.Array:
        key = jax.random.PRNGKey(key)
        return self._predict(self._state.variables, key, image_indices)

    def encode(self, key: int | None, images: jax.Array) -> jax.Array:
        return self._encode(self._state.variables, key, images)

    def _encode(
        self,
        model_variables: ModelVariables,
        key: jax.random.KeyArray | None,
        images: jax.Array,
    ) -> jax.Array:
        latent_mean, latent_log_variance = self._encoder.apply(
            model_variables.encoder, images
        )
        if key is None:
            sampled_latent = latent_mean
        else:
            key = jax.random.PRNGKey(key)
            sampled_latent = latent_mean + jnp.exp(
                latent_log_variance * 0.5
            ) * jax.random.normal(key, latent_log_variance.shape)
        return sampled_latent

    def _predict(
        self,
        model_variables: ModelVariables,
        key: jax.random.PRNGKey,
        images: jax.Array,
    ) -> jax.Array:
        latent_mean, latent_log_variance = self._encoder.apply(
            model_variables.encoder, images
        )
        sampled_latent = latent_mean + jnp.exp(
            latent_log_variance * 0.5
        ) * jax.random.normal(key, latent_log_variance.shape)
        predictions = self._decoder.apply(model_variables.decoder, sampled_latent)
        predictions = 255.0 * nn.sigmoid(predictions)
        return predictions
