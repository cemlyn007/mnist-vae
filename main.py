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
import PIL.Image

T = TypeVar("T")


def measure_duration_ms(func: Callable[..., T], *args, **kwargs) -> tuple[T, float]:
    start_time = time.monotonic()
    result = func(*args, **kwargs)
    end_time = time.monotonic()
    elapsed_time = end_time - start_time
    return result, elapsed_time * 1000.0


def predict_images(
    this_experiment: experiment.Experiment, image_ids: list[int]
) -> list[PIL.Image.Image]:
    images = this_experiment.predict(0, image_ids)
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


def get_hyperparameters(last_neptune_run: str) -> experiment.Hyperparameters:
    read_logger = logger.Logger(
        neptune_project, neptune_api_token, last_neptune_run, read_only=True
    )
    try:
        hyperparameters = experiment.Hyperparameters(
            latent_dims=read_logger.get_int("hyperparameters/latent_dims"),
            learning_rate=read_logger.get_float("hyperparameters/learning_rate"),
        )
    except:
        read_logger.close()
    return hyperparameters


def get_last_hyperparameters_and_settings(
    last_neptune_run: str,
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
            learning_rate=read_logger.get_last_float("metrics/learning_rate"),
            beta=read_logger.get_last_float("metrics/beta"),
            batch_size=read_logger.get_last_int("metrics/batch_size"),
            predict_interval=predict_interval,
            tsne_interval=tsne_interval,
            tsne_perplexity=tsne_perplexity,
            tsne_iterations=tsne_iterations,
            checkpoint_interval=checkpoint_interval,
            checkpoint_max_to_keep=checkpoint_max_to_keep,
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


if __name__ == "__main__":
    import os
    import numpy as np
    import argparse
    import shutil
    import platform
    import math

    PREDICT_IMAGE_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    TSNE_IMAGE_IDS = list(range(3072))

    default_experiment_directory = os.path.join(os.getcwd(), "experiments")
    if platform.system() == "Darwin":
        default_experiment_directory = default_experiment_directory + ".nosync"

    parser = argparse.ArgumentParser("MNIST VAE")
    parser.add_argument("--new_experiment", action="store_true")
    parser.add_argument("--backend", type=str, default=None)
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
    batch_size: int = args.batch_size
    predict_interval: int = args.predict_interval
    tsne_interval: int = args.tsne_interval
    tsne_perplexity: float = args.tsne_perplexity
    tsne_iterations: int = args.tsne_iterations
    experiment_directory: str = args.experiment_directory
    checkpoint_save_interval: int = args.checkpoint_save_interval
    checkpoint_max_to_keep: int = args.checkpoint_max_to_keep
    neptune_project: str | None = args.neptune_project
    neptune_api_token: str | None = args.neptune_api_token

    cache_directory = os.path.join(experiment_directory, "cache")
    checkpoint_path = os.path.join(experiment_directory, "checkpoints")
    last_neptune_run_path = os.path.join(experiment_directory, ".last_neptune_run")

    if new_experiment:
        if os.path.exists(last_neptune_run_path):
            os.remove(last_neptune_run_path)
        shutil.rmtree(experiment_directory, ignore_errors=True)
        last_neptune_run = None
        hyperparameters = experiment.Hyperparameters(latent_dims=16, learning_rate=1e-3)
        settings = renderer.Settings(
            beta=0.5,
            learning_rate=hyperparameters.learning_rate,
            predict_interval=predict_interval,
            tsne_interval=tsne_interval,
            tsne_perplexity=tsne_perplexity,
            tsne_iterations=tsne_iterations,
            batch_size=batch_size,
            checkpoint_interval=checkpoint_save_interval,
            checkpoint_max_to_keep=checkpoint_max_to_keep,
        )
    else:
        last_neptune_run = get_last_neptune_run(last_neptune_run_path)
        hyperparameters, settings = get_last_hyperparameters_and_settings(
            last_neptune_run,
            predict_interval=predict_interval,
            tsne_interval=tsne_interval,
            tsne_perplexity=tsne_perplexity,
            tsne_iterations=tsne_iterations,
            checkpoint_interval=checkpoint_save_interval,
            checkpoint_max_to_keep=checkpoint_max_to_keep,
        )

    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)

    view = renderer.Renderer(settings, os.path.join(os.getcwd(), "assets", "icon.png"))
    try:
        write_logger = logger.Logger(
            neptune_project, neptune_api_token, last_neptune_run, flush_period=1.0
        )
        try:
            if new_experiment:
                cache_neptune_run_id(last_neptune_run_path, write_logger.run_id)

            this_experiment = experiment.Experiment(
                hyperparameters, checkpoint_path, cache_directory, backend
            )
            try:
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
                            predict_images, this_experiment, PREDICT_IMAGE_IDS
                        )
                        log_images.update(
                            {
                                f"train/predicted_images/{image_id}": image
                                for image_id, image in zip(
                                    PREDICT_IMAGE_IDS, predicted_images, strict=True
                                )
                            }
                        )

                    if settings.tsne_interval and step % settings.tsne_interval == 0:
                        labels = this_experiment.train_labels[jnp.array(TSNE_IMAGE_IDS)]
                        (
                            latent_samples,
                            log_values["profile/encode_duration_ms"],
                        ) = measure_duration_ms(
                            this_experiment.encode,
                            None,
                            TSNE_IMAGE_IDS,
                        )
                        (
                            log_images["train/embeddings"],
                            log_values["profile/tsne_duration_ms"],
                        ) = measure_duration_ms(
                            get_tsne_plot,
                            latent_samples,
                            labels,
                            settings.tsne_perplexity,
                            settings.tsne_iterations,
                        )

                    write_logger.append_values(log_values, step, timestamp)
                    log_values.clear()
                    write_logger.append_images(log_images, step, timestamp)
                    log_images.clear()
                    end_iteration = time.monotonic()
                    iteration_duration_ms = (end_iteration - start_iteration) * 1000.0
                    print(f"Step {step} took {iteration_duration_ms:.2f}ms")

            except KeyboardInterrupt:
                print("Stopping...", flush=True)
            finally:
                this_experiment.close()
        finally:
            write_logger.close()
    finally:
        view.close()
