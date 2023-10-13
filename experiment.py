from flax import linen as nn
import jax
import typing
import mnist
import jax.numpy as jnp
import jax
import optax
import orbax.checkpoint as ocp
import functools
import operator


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
        hyperparameters: Hyperparameters,
        checkpoint_directory: str,
        cache_directory: str,
        backend: str | None = None,
    ) -> None:
        self._device = jax.devices(backend)[0]
        self._checkpoint_manager = self._get_checkpoint_manager(checkpoint_directory)
        self._train_images, self.train_labels = self._get_training_data(cache_directory)
        self._encoder = self._get_encoder(hyperparameters.latent_dims)
        self._decoder = self._get_decoder()
        self._optimizer = self._get_optimizer(hyperparameters.learning_rate)

        self._train_step = jax.jit(self._train_step, device=self._device)
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
            step = self._state.step.item()
            if self._checkpoint_manager.latest_step() != step:
                self._checkpoint_manager.save(
                    self._state.step.item(), self._state, force=True
                )
        finally:
            self._checkpoint_manager.close()

    def checkpoint_exists(self) -> bool:
        return self._checkpoint_manager.latest_step() is not None

    def _get_checkpoint_manager(
        self, checkpoint_directory: str
    ) -> ocp.CheckpointManager:
        options = ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=100)
        checkpoint_manager = ocp.CheckpointManager(
            checkpoint_directory, ocp.PyTreeCheckpointer(), options=options
        )
        return checkpoint_manager

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

    def _get_encoder(self, latent_dims: int) -> Encoder:
        return Encoder(latent_dims=latent_dims)

    def _get_decoder(self) -> Decoder:
        return Decoder()

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

    def train_step(self, learning_rate: float, beta: float) -> dict[str, any]:
        learning_rate = jnp.float32(learning_rate)
        if learning_rate != self._state.optimizer_state.hyperparams["learning_rate"]:
            self._optimizer = self._get_optimizer(learning_rate)
            self._state = self._state._replace(
                optimizer_state=self._optimizer.init(self._state.variables)
            )
        self._state, metrics = self._train_step(self._state, beta)
        step = self._state.step.item()
        self._checkpoint_manager.save(step, self._state, metrics=metrics)
        return {
            "step": step,
            **jax.tree_map(
                lambda x: jnp.finfo(metrics["learning_rate"].dtype).max.item()
                if jnp.isinf(x)
                else x.item(),
                metrics,
            ),
        }

    def _train_step(self, state: State, beta: float) -> tuple[State, dict[str, any]]:
        new_key, key = jax.random.split(state.key)
        (loss, extras), grad = jax.value_and_grad(self._get_loss, has_aux=True)(
            state.variables, key, beta
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
            "learning_rate": new_optimizer_state.hyperparams["learning_rate"],
            **extras,
        }

    def _get_loss(
        self, model_variables: ModelVariables, key: jax.random.PRNGKey, beta: float
    ) -> tuple[jax.Array, dict[str, any]]:
        sample_key, key = jax.random.split(key)
        train_images = self._train_images[
            jax.random.randint(sample_key, (8192,), 0, self._train_images.shape[0] + 1)
        ]
        latent_mean, latent_log_variance = self._encoder.apply(
            model_variables.encoder, train_images
        )
        sampled_latent = latent_mean + jnp.exp(
            latent_log_variance * 0.5
        ) * jax.random.normal(key, latent_log_variance.shape)
        predictions = self._decoder.apply(model_variables.decoder, sampled_latent)

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

    def predict(self, key: int, image_indices: list[int]) -> jax.Array:
        key = jax.random.PRNGKey(key)
        return self._predict(self._state.variables, key, image_indices)

    def encode(self, key: int | None, image_indices: list[int]) -> jax.Array:
        return self._encode(self._state.variables, key, image_indices)

    def _encode(
        self,
        model_variables: ModelVariables,
        key: jax.random.KeyArray | None,
        image_indices: list[int],
    ) -> jax.Array:
        train_images = self._train_images[jnp.array(image_indices)]
        latent_mean, latent_log_variance = self._encoder.apply(
            model_variables.encoder, train_images
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
        image_indices: list[int],
    ) -> jax.Array:
        image_indices = jnp.array(image_indices)
        latent_mean, latent_log_variance = self._encoder.apply(
            model_variables.encoder, self._train_images[image_indices]
        )
        sampled_latent = latent_mean + jnp.exp(
            latent_log_variance * 0.5
        ) * jax.random.normal(key, latent_log_variance.shape)
        predictions = self._decoder.apply(model_variables.decoder, sampled_latent)
        predictions = 255.0 * nn.sigmoid(predictions)
        return predictions
