from flax import linen as nn
import jax
import typing
import os
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
        self._train_images = self._get_train_images(cache_directory)
        self._encoder = self._get_encoder(hyperparameters.latent_dims)
        self._decoder = self._get_decoder()
        self._optimizer = self._get_optimizer(hyperparameters.learning_rate)

        self._train_step = jax.jit(self._train_step, device=self._device)
        self._predict = jax.jit(self._predict, device=self._device)

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

    def _get_train_images(self, cache_directory: str) -> jax.Array:
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
        return train_images

    def _get_encoder(self, latent_dims: int) -> Encoder:
        return Encoder(latent_dims=latent_dims)

    def _get_decoder(self) -> Decoder:
        return Decoder()

    def _get_optimizer(self, learning_rate: float) -> optax.GradientTransformation:
        # return optax.inject_hyperparams(optax.adamaxw)(learning_rate=learning_rate)
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

    def train_step(self) -> dict[str, any]:
        self._state, metrics = self._train_step(self._state)
        step = self._state.step.item()
        self._checkpoint_manager.save(step, self._state, metrics=metrics)
        return {"step": step, **metrics}

    def _train_step(self, state: State) -> tuple[State, dict[str, any]]:
        new_key, key = jax.random.split(state.key)
        beta = 0.025
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


if __name__ == "__main__":
    import time
    import neptune
    import numpy as np
    import argparse
    import PIL.Image
    import shutil
    import platform

    default_experiment_directory = os.path.join(os.getcwd(), "experiments")
    if platform.system() == "Darwin":
        default_experiment_directory = default_experiment_directory + ".nosync"

    parser = argparse.ArgumentParser("MNIST VAE")
    parser.add_argument("--new_experiment", action="store_true")
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--predict_interval", type=int, default=0, help="0 to disable")
    parser.add_argument(
        "--experiment_directory",
        type=str,
        default=default_experiment_directory,
        help='On macOS you might want to append ".nosync" to the folder name to prevent iCloud syncing it.',
    )
    parser.add_argument(
        "--neptune_project",
        type=str,
        default=None,
        help='If not specified, will fallback to using the environment variable "NEPTUNE_PROJECT"',
    )
    parser.add_argument(
        "--neptune_api_token",
        type=str,
        default=None,
        help='If not specified, will fallback to using the environment variable "NEPTUNE_API_TOKEN"',
    )
    args = parser.parse_args()
    new_experiment: bool = args.new_experiment
    backend: str | None = args.backend
    predict_interval: int = args.predict_interval
    experiment_directory: str = args.experiment_directory
    neptune_project: str | None = args.neptune_project
    neptune_api_token: str | None = args.neptune_api_token

    cache_directory = os.path.join(experiment_directory, "cache")
    checkpoint_path = os.path.join(experiment_directory, "checkpoints")
    last_neptune_run_path = os.path.join(experiment_directory, ".last_neptune_run")

    if new_experiment:
        if os.path.exists(last_neptune_run_path):
            os.remove(last_neptune_run_path)
        shutil.rmtree(experiment_directory, ignore_errors=True)
        hyperparameters = Hyperparameters(latent_dims=16, learning_rate=1e-3)
    else:
        if os.path.exists(last_neptune_run_path):
            with open(last_neptune_run_path, "r") as f:
                last_neptune_run = f.read()
            run = neptune.init_run(
                with_id=last_neptune_run,
                project=neptune_project,
                api_token=neptune_api_token,
                capture_hardware_metrics=False,
                capture_stderr=False,
                capture_stdout=False,
                capture_traceback=False,
                mode="read-only",
            )
            try:
                hyperparameters = Hyperparameters(
                    latent_dims=run["parameters/latent_dims"].fetch(),
                    learning_rate=run["parameters/learning_rate"].fetch(),
                )
            finally:
                run.stop()
        else:
            raise RuntimeError("No last Neptune run found!")

    experiment = Experiment(hyperparameters, checkpoint_path, cache_directory, backend)
    try:
        if experiment.checkpoint_exists():
            experiment.restore()
        else:
            last_neptune_run = None
            experiment.reset(0)

        run = neptune.init_run(
            with_id=last_neptune_run,
            project=neptune_project,
            api_token=neptune_api_token,
            capture_hardware_metrics=False,
            capture_stderr=False,
            capture_stdout=False,
            capture_traceback=False,
            flush_period=10.0,
        )
        try:
            with open(last_neptune_run_path, "w") as f:
                f.write(run["sys/id"].fetch())

            # TODO: I am undecided here because I would like to
            run["parameters"] = hyperparameters._asdict()
            for key, value in hyperparameters._asdict().items():
                run[f"hyperparameters/{key}"] = value

            profile = {}
            while True:
                profile.clear()
                start_train_step = time.monotonic()
                metrics = experiment.train_step()
                end_train_step = time.monotonic()
                train_step_duration = end_train_step - start_train_step
                profile["train_step_duration_ms"] = train_step_duration * 1000.0
                timestamp = time.time()
                step = metrics.pop("step")

                for key, value in metrics.items():
                    run[f"metrics/{key}"].append(value, timestamp=timestamp, step=step)

                # Do this to be less memory hoggy.
                if predict_interval and step % predict_interval == 0:
                    start_predict = time.monotonic()
                    image_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                    predicted_images = experiment.predict(0, image_ids)
                    predicted_images = np.reshape(
                        np.round(np.asarray(predicted_images, dtype=np.float32)),
                        (-1, 28, 28),
                    ).astype(np.uint8)
                    end_predict = time.monotonic()
                    predict_duration = end_predict - start_predict
                    profile["predict_duration_ms"] = predict_duration * 1000.0
                    start_create_figures = time.monotonic()
                    for image_id, predicted_image in zip(image_ids, predicted_images):
                        image = PIL.Image.fromarray(predicted_image)
                        run[f"train/distribution/{image_id}"].append(
                            image,
                            timestamp=timestamp,
                            step=step,
                        )
                    end_create_figures = time.monotonic()
                    create_figures_duration = end_create_figures - start_create_figures
                    profile["create_figures_duration_ms"] = (
                        create_figures_duration * 1000.0
                    )

                for key, value in profile.items():
                    run[f"profile/{key}"].append(value, timestamp=timestamp, step=step)

        except KeyboardInterrupt:
            print("Stopping...", flush=True)
        finally:
            run.stop()
    finally:
        experiment.close()
