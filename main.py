import numpy as np
import PIL.Image
from typing import Callable, TypeVar
import experiment
import time
import logger
import renderer
import jax
import tsne
from matplotlib.backends import backend_agg
import matplotlib.figure
import matplotlib.cm
import matplotlib.patches
import jax.numpy as jnp
import sys

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


def get_last_neptune_run(last_neptune_file_path: str) -> str:
    with open(last_neptune_file_path, "r") as f:
        return f.read()


def cache_neptune_run_id(last_neptune_file_path: str, run_id: str) -> None:
    with open(last_neptune_file_path, "w") as f:
        f.write(run_id)


def get_last_hyperparameters_and_settings(
    last_neptune_run: str,
    neptune_project: str,
    neptune_api_token: str,
    predict_interval: int,
    tsne_interval: int,
    tsne_perplexity: int,
    tsne_iterations: int,
    checkpoint_interval: int,
    checkpoint_max_to_keep: int,
) -> tuple[experiment.Hyperparameters, renderer.Settings]:
    read_logger = logger.Logger(
        neptune_project, neptune_api_token, last_neptune_run, read_only=True
    )
    try:
        hyperparameters = experiment.Hyperparameters(
            latent_dims=read_logger.get_int("hyperparameters/latent_dims"),
            learning_rate=read_logger.get_float("hyperparameters/learning_rate"),
        )
        settings = renderer.Settings(
            latent_size=hyperparameters.latent_dims,
            learning_rate=read_logger.get_last_float("metrics/learning_rate"),
            beta=read_logger.get_last_float("metrics/beta"),
            batch_size=read_logger.get_last_int("metrics/batch_size"),
            predict_interval=predict_interval,
            tsne_interval=tsne_interval,
            tsne_perplexity=tsne_perplexity,
            tsne_iterations=tsne_iterations,
            checkpoint_interval=checkpoint_interval,
            checkpoint_max_to_keep=checkpoint_max_to_keep,
            neptune_project_name=neptune_project,
            neptune_api_token=neptune_api_token,
            state=renderer.State.PAUSED,
        )
    finally:
        read_logger.close()
    return hyperparameters, settings


estimate_tsne = jax.jit(tsne.estimate_tsne)


def get_tsne_plot(
    latent_samples: jax.Array, labels: jax.Array, perplexity: float, iterations: int
) -> PIL.Image.Image:
    embeddings = estimate_tsne(
        latent_samples,
        jax.random.PRNGKey(0),
        perplexity=perplexity,
        iterations=iterations,
        learning_rate=10.0,
        momentum=0.9,
    )
    color_map = matplotlib.cm.rainbow(np.linspace(0, 1, 10))
    DPI = 200
    fig = matplotlib.figure.Figure(figsize=(1080 / DPI, 1080 / DPI), dpi=DPI)
    ax = fig.add_subplot(111)
    ax.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=color_map[labels],
        alpha=0.1,
    )
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.legend(
        handles=[
            matplotlib.patches.Patch(color=c, label=str(i))
            for i, c in enumerate(color_map)
        ],
        loc="upper right",
    )
    fig.tight_layout()
    canvas = backend_agg.FigureCanvasAgg(fig)
    image_bytes, size = canvas.print_to_buffer()
    return PIL.Image.frombuffer("RGBA", size, image_bytes)


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


if __name__ == "__main__":
    import os
    import numpy as np
    import argparse
    import shutil
    import platform
    import math
    import keyring
    import json
    import webbrowser

    get_some_images = jax.jit(
        get_some_images,
        static_argnames=("images_per_digit",),
        backend="cpu" if sys.platform == "darwin" else None,
    )

    default_experiment_directory = os.path.join(os.getcwd(), "experiments")
    if platform.system() == "Darwin":
        default_experiment_directory = default_experiment_directory + ".nosync"

    parser = argparse.ArgumentParser("MNIST VAE")
    parser.add_argument("--resume_experiment", action="store_true")
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--latent_size", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--predict_interval", type=int, default=0, help="0 to disable")
    parser.add_argument("--tsne_interval", type=int, default=0, help="0 to disable")
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_iterations", type=int, default=1000)
    parser.add_argument(
        "--experiment_directory",
        type=str,
        default=default_experiment_directory,
        help='On macOS you might want to append ".nosync" to the folder name to prevent iCloud syncing it.',
    )
    parser.add_argument(
        "--checkpoint_save_interval",
        type=int,
        default=0,
        help="Disable by setting to 0.",
    )
    parser.add_argument(
        "--checkpoint_max_to_keep",
        type=int,
        default=0,
        help="Disable by setting to 0.",
    )
    parser.add_argument(
        "--neptune_project",
        type=str,
        default="",
        help='If not specified, will fallback to using the environment variable "NEPTUNE_PROJECT"',
    )
    parser.add_argument(
        "--neptune_api_token",
        type=str,
        default="",
        help='If not specified, will fallback to using the environment variable "NEPTUNE_API_TOKEN"',
    )
    args = parser.parse_args()
    new_experiment: bool = not args.resume_experiment
    backend: str | None = args.backend
    latent_size: int = args.latent_size
    batch_size: int = args.batch_size
    predict_interval: int = args.predict_interval
    tsne_interval: int = args.tsne_interval
    tsne_perplexity: float = args.tsne_perplexity
    tsne_iterations: int = args.tsne_iterations
    experiment_directory: str = args.experiment_directory
    checkpoint_save_interval: int = args.checkpoint_save_interval
    checkpoint_max_to_keep: int = args.checkpoint_max_to_keep
    neptune_project: str = args.neptune_project
    neptune_api_token: str = args.neptune_api_token

    cache_directory = os.path.join(experiment_directory, "cache")
    checkpoint_path = os.path.join(experiment_directory, "checkpoints")
    last_neptune_run_path = os.path.join(experiment_directory, ".last_neptune_run")
    if "NEPTUNE_DATA_DIRECTORY" not in os.environ:
        os.environ["NEPTUNE_DATA_DIRECTORY"] = os.path.join(
            experiment_directory, "neptune"
        )

    credential = keyring.get_credential("mnist-vae", "")
    if credential:
        password = json.loads(credential.password)
        if not neptune_project:
            neptune_project = password["neptune_project_name"]
        if not neptune_api_token:
            neptune_api_token = password["neptune_api_token"]

    if new_experiment:
        if os.path.exists(last_neptune_run_path):
            os.remove(last_neptune_run_path)
        shutil.rmtree(experiment_directory, ignore_errors=True)
        last_neptune_run = None
        hyperparameters = experiment.Hyperparameters(
            latent_dims=latent_size, learning_rate=1e-3
        )
        settings = renderer.Settings(
            latent_size=latent_size,
            beta=0.5,
            learning_rate=hyperparameters.learning_rate,
            predict_interval=predict_interval,
            tsne_interval=tsne_interval,
            tsne_perplexity=tsne_perplexity,
            tsne_iterations=tsne_iterations,
            batch_size=batch_size,
            checkpoint_interval=checkpoint_save_interval,
            checkpoint_max_to_keep=checkpoint_max_to_keep,
            neptune_project_name=neptune_project,
            neptune_api_token=neptune_api_token,
            state=renderer.State.NEW,
        )
    else:
        last_neptune_run = get_last_neptune_run(last_neptune_run_path)
        hyperparameters, settings = get_last_hyperparameters_and_settings(
            last_neptune_run,
            neptune_project,
            neptune_api_token,
            predict_interval=predict_interval,
            tsne_interval=tsne_interval,
            tsne_perplexity=tsne_perplexity,
            tsne_iterations=tsne_iterations,
            checkpoint_interval=checkpoint_save_interval,
            checkpoint_max_to_keep=checkpoint_max_to_keep,
        )

    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)

    view = renderer.Renderer(
        settings, os.path.join(os.path.dirname(__file__), "assets", "icon.png")
    )
    try:
        while view.open and settings.state != renderer.State.RUNNING:
            settings = view.update()
            time.sleep(1.0 / 30.0)
        if view.open:
            credential = keyring.get_credential("mnist-vae", "")
            if credential is None:
                if settings.neptune_project_name is None:
                    raise ValueError(
                        "Neptune project name not specified and not found in keyring."
                    )
                if settings.neptune_api_token is None:
                    raise ValueError(
                        "Neptune API token not specified and not found in keyring."
                    )
                # else...
                keyring.set_password(
                    "mnist-vae",
                    "",
                    json.dumps(
                        {
                            "neptune_project_name": settings.neptune_project_name,
                            "neptune_api_token": settings.neptune_api_token,
                        }
                    ),
                )
            else:
                # The credentials exist, but they might be outdated.
                password = json.loads(credential.password)
                save_password = False
                if (
                    settings.neptune_project_name
                    and password["neptune_project_name"]
                    != settings.neptune_project_name
                ):
                    save_password = True
                    password["neptune_project_name"] = settings.neptune_project_name
                else:
                    settings = settings._replace(
                        neptune_project_name=password["neptune_project_name"]
                    )
                if (
                    settings.neptune_api_token
                    and credential.password != settings.neptune_api_token
                ):
                    save_password = True
                    password["neptune_api_token"] = settings.neptune_api_token
                else:
                    settings = settings._replace(
                        neptune_api_token=password["neptune_api_token"]
                    )

                if save_password:
                    keyring.set_password("mnist-vae", "", json.dumps(password))

            write_logger = logger.Logger(
                settings.neptune_project_name,
                settings.neptune_api_token,
                last_neptune_run,
                flush_period=1.0,
            )
            try:
                url = write_logger.get_url()
                webbrowser.open_new_tab(url)
                if new_experiment:
                    cache_neptune_run_id(last_neptune_run_path, write_logger.run_id)

                this_experiment = experiment.Experiment(
                    hyperparameters, checkpoint_path, cache_directory, backend
                )
                try:
                    (
                        images_to_encode_decode,
                        _,
                    ) = get_some_images(
                        this_experiment.train_images, this_experiment.train_labels, 1
                    )
                    tsne_images, tsne_labels = get_some_images(
                        this_experiment.train_images, this_experiment.train_labels, 256
                    )

                    if this_experiment.checkpoint_exists():
                        this_experiment.restore()
                    else:
                        this_experiment.reset(0)

                    write_logger.set_values(
                        {
                            f"hyperparameters/{key}": value
                            for key, value in hyperparameters._asdict().items()
                        }
                    )

                    log_values = {}
                    log_images = {}
                    while view.open:
                        start_iteration = time.monotonic()
                        settings = view.update()

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
                                    raise ValueError(f"NaN for key {key}")
                                elif math.isinf(value):
                                    raise ValueError(f"Inf for key {key}")
                                elif not math.isfinite(value):
                                    raise ValueError(f"Non-finite for key {key}")

                            log_values.update(
                                {
                                    f"metrics/{key}": value
                                    for key, value in metrics.items()
                                }
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
                                        for digit_id, image in enumerate(
                                            predicted_images
                                        )
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
                                    get_tsne_plot,
                                    latent_samples,
                                    tsne_labels,
                                    settings.tsne_perplexity,
                                    settings.tsne_iterations,
                                )

                            write_logger.append_values(log_values, step, timestamp)
                            log_values.clear()
                            write_logger.append_images(log_images, step, timestamp)
                            log_images.clear()
                        elif settings.state == renderer.State.PAUSED:
                            time.sleep(1.0 / 30.0)
                        end_iteration = time.monotonic()
                        iteration_duration_ms = (
                            end_iteration - start_iteration
                        ) * 1000.0
                        print(
                            f"Step {step} with state {settings.state} took {iteration_duration_ms:.2f}ms"
                        )

                except KeyboardInterrupt:
                    print("Stopping...", flush=True)
                finally:
                    this_experiment.close()
            finally:
                write_logger.close()
    finally:
        view.close()
