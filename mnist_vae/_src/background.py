from typing import Callable, TypeVar
import numpy as np
import PIL.Image
from mnist_vae import experiment, renderer, tsne
import time
import jax
from matplotlib.backends import backend_agg
import matplotlib.figure
import matplotlib.cm
import matplotlib.patches
import jax.numpy as jnp
import sys
import math
import multiprocessing.connection
import traceback
import importlib.util


class BadUpdateError(ValueError):
    pass


T = TypeVar("T")


def measure_duration_ms(func: Callable[..., T], *args, **kwargs) -> tuple[T, float]:
    start_time = time.monotonic()
    result = func(*args, **kwargs)
    end_time = time.monotonic()
    elapsed_time = end_time - start_time
    return result, elapsed_time * 1000.0


def encode_decode_images(
    this_experiment: experiment.Experiment, images: jax.Array
) -> list[PIL.Image.Image]:
    images = this_experiment.encode_decode(0, images)
    images = np.reshape(
        np.round(np.asarray(images, dtype=np.float32)),
        (-1, 28, 28),
    ).astype(np.uint8)
    return [PIL.Image.fromarray(image) for image in images]


def get_some_images(
    images: jax.Array, labels: jax.Array, images_per_digit: int
) -> tuple[jax.Array, jax.Array]:
    flattened_images = []
    flattened_labels = []
    for i in range(10):
        indices = jnp.argwhere(labels == i, size=images_per_digit).flatten()
        flattened_images.append(images[indices])
        flattened_labels.append(labels[indices])

    flattened_images = jnp.concatenate(flattened_images, axis=0)
    flattened_labels = jnp.concatenate(flattened_labels, axis=0)
    return flattened_images, flattened_labels


estimate_tsne = jax.jit(tsne.estimate_tsne)

get_some_images = jax.jit(
    get_some_images,
    static_argnames=("images_per_digit",),
    backend="cpu" if sys.platform == "darwin" else None,
)


class MnistTsneRenderer:
    def __init__(self, backend: str) -> None:
        self._image_width = 720
        self._image_height = 720
        self._padding_proportion = 0.25
        self._point_radius = 3
        self._color_map = matplotlib.cm.rainbow(np.linspace(0, 1, 10))
        self._fig = matplotlib.figure.Figure(figsize=(6, 6), dpi=200, layout="tight")
        self._fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        self._ax = self._fig.add_axes((0.0, 0.0, 1.0, 1.0))
        self._ax.legend(
            handles=[
                matplotlib.patches.Patch(color=c, label=str(i))
                for i, c in enumerate(self._color_map)
            ],
            loc="upper right",
            borderaxespad=0.0,
        )
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self._ax.get_xaxis().set_visible(False)
        self._ax.get_yaxis().set_visible(False)
        self._ax.autoscale(enable=True, axis="both")
        self._canvas = backend_agg.FigureCanvasAgg(self._fig)
        self._get_points = jax.jit(self._get_points, device=jax.devices(backend)[0])

    def __call__(
        self,
        latent_samples: jax.Array,
        labels: jax.Array,
        perplexity: float,
        iterations: int,
    ) -> PIL.Image.Image:
        points = self._get_points(
            latent_samples,
            perplexity=perplexity,
            iterations=iterations,
        )
        points = np.asarray(points)
        scatter = self._ax.scatter(
            points[:, 0],
            points[:, 1],
            c=self._color_map[labels],
            alpha=0.1,
        )
        try:
            image_bytes, size = self._canvas.print_to_buffer()
        finally:
            scatter.remove()
        image = PIL.Image.frombuffer("RGBA", size, image_bytes)
        return image

    def _get_points(
        self,
        latent_samples: jax.Array,
        perplexity: float,
        iterations: int,
    ) -> jax.Array:
        embeddings = tsne.estimate_tsne(
            latent_samples,
            jax.random.PRNGKey(0),
            perplexity=perplexity,
            iterations=iterations,
            learning_rate=10.0,
            momentum=0.9,
        )
        return embeddings


def get_some_images(
    images: jax.Array, labels: jax.Array, images_per_digit: int
) -> tuple[jax.Array, jax.Array]:
    flattened_images = []
    flattened_labels = []

    if labels.device().platform == "METAL":
        device = jax.devices("cpu")[0]
    else:
        device = labels.device()

    for i in range(10):
        indices = jnp.argwhere(
            jax.device_put(labels, device) == i, size=images_per_digit
        ).flatten()
        indices = jax.device_put(indices, images.device())
        flattened_images.append(images[indices])
        flattened_labels.append(labels[indices])

    flattened_images = jnp.concatenate(flattened_images, axis=0)
    flattened_labels = jnp.concatenate(flattened_labels, axis=0)
    return flattened_images, flattened_labels


class UserModels:
    def __init__(self, model_filepath: str) -> None:
        spec = importlib.util.spec_from_file_location("user_data", model_filepath)
        self._user_data = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._user_data)

    def get_encoder(
        self, latent_dims: int
    ) -> Callable[[jax.Array], experiment.Encoder]:
        return self._user_data.Encoder(latent_dims)

    def get_decoder(self) -> Callable[[], experiment.Decoder]:
        return self._user_data.Decoder()


def experiment_process(
    connection: multiprocessing.connection.Connection,
    hyperparameters: experiment.Hyperparameters,
    checkpoint_path: str,
    cache_directory: str,
    backend: str | None,
) -> None:
    try:
        try:
            connection.send("ready")
            settings = connection.recv()
            if not isinstance(settings, renderer.Settings):
                raise TypeError(
                    f"Expected renderer.Settings, got {type(settings)}: {settings}"
                )
            # else...

            mnst_tsne_renderer = MnistTsneRenderer(backend)

            user_models = UserModels(settings.model_filepath)

            this_experiment = experiment.Experiment(
                user_models.get_encoder,
                user_models.get_decoder,
                hyperparameters,
                checkpoint_path,
                cache_directory,
                backend,
            )
            try:
                (
                    images_to_encode_decode,
                    _,
                ) = get_some_images(
                    this_experiment.train_images,
                    this_experiment.train_labels,
                    1,
                )
                tsne_images, tsne_labels = get_some_images(
                    this_experiment.train_images,
                    this_experiment.train_labels,
                    256,
                )

                if this_experiment.checkpoint_exists():
                    this_experiment.restore()
                else:
                    this_experiment.reset(0)

                log_values = {}
                log_images = {}

                step = 0
                while settings.state != renderer.State.NEW and not connection.closed:
                    start_iteration = time.monotonic()

                    if connection.poll(0.0):
                        try:
                            message = connection.recv()
                            if message == "kill":
                                return
                            settings = message
                        except EOFError:
                            return

                    if settings.state == renderer.State.RUNNING:
                        if (
                            settings.checkpoint_interval
                            != this_experiment.checkpoint_interval
                            or settings.checkpoint_max_to_keep
                            != this_experiment.checkpoint_max_to_keep
                        ):
                            this_experiment.update_checkpoint_manager(
                                settings.checkpoint_interval,
                                settings.checkpoint_max_to_keep,
                            )

                        (
                            metrics,
                            log_values["profile/train_step_duration_ms"],
                        ) = measure_duration_ms(
                            this_experiment.train_step,
                            settings.learning_rate,
                            settings.beta,
                            settings.batch_size,
                        )

                        for key, value in metrics.items():
                            if not isinstance(value, (float, int)):
                                raise TypeError(
                                    f"Expected float or int, got {type(value)} for key {key}"
                                )
                            elif math.isnan(value):
                                raise BadUpdateError(f"NaN for key {key}")
                            elif math.isinf(value):
                                raise BadUpdateError(f"Inf for key {key}")
                            elif not math.isfinite(value):
                                raise BadUpdateError(f"Non-finite for key {key}")

                        log_values.update(
                            {f"metrics/{key}": value for key, value in metrics.items()}
                        )

                        timestamp = time.time()
                        step: int = metrics.pop("step")

                        if (
                            settings.predict_interval
                            and step % settings.predict_interval == 0
                        ):
                            (
                                predicted_images,
                                log_values["profile/predict_duration_ms"],
                            ) = measure_duration_ms(
                                encode_decode_images,
                                this_experiment,
                                images_to_encode_decode,
                            )
                            log_images.update(
                                {
                                    f"train/predicted_images/{digit_id}": image
                                    for digit_id, image in enumerate(predicted_images)
                                }
                            )

                        if (
                            settings.tsne_interval
                            and step % settings.tsne_interval == 0
                        ):
                            (
                                latent_samples,
                                log_values["profile/encode_duration_ms"],
                            ) = measure_duration_ms(
                                this_experiment.encode,
                                None,
                                tsne_images,
                            )
                            (
                                log_images["train/embeddings"],
                                log_values["profile/tsne_duration_ms"],
                            ) = measure_duration_ms(
                                mnst_tsne_renderer,
                                latent_samples,
                                tsne_labels,
                                settings.tsne_perplexity,
                                settings.tsne_iterations,
                            )

                        connection.send(
                            (
                                step,
                                timestamp,
                                log_values,
                                log_images,
                            )
                        )
                        log_values.clear()
                        log_images.clear()
                    elif settings.state == renderer.State.PAUSED:
                        time.sleep(1.0 / 30.0)
                    end_iteration = time.monotonic()
                    iteration_duration_ms = (end_iteration - start_iteration) * 1000.0
                    print(
                        f"Step {step} with state {settings.state} took {iteration_duration_ms:.2f}ms"
                    )
            except (KeyboardInterrupt, BrokenPipeError, EOFError) as exception:
                print("Process got:", exception, flush=True)
            finally:
                this_experiment.close()
        finally:
            connection.close()
    except Exception as exception:
        print("Exception in child", exception, traceback.format_exc(), flush=True)
