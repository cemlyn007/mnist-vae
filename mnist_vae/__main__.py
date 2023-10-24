from mnist_vae import experiment, logger, renderer
import time
import sys
import neptune.common.exceptions
from mnist_vae._src import background


def get_last_neptune_run(last_neptune_file_path: str) -> str:
    with open(last_neptune_file_path, "r") as f:
        return f.read()


def cache_neptune_run_id(last_neptune_file_path: str, run_id: str) -> None:
    with open(last_neptune_file_path, "w") as f:
        f.write(run_id)


def get_last_hyperparameters_and_settings(
    experiment_directory: str,
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
        model_filepath = os.path.join(experiment_directory, "model.py")
        read_logger.download_file("model", experiment_directory)
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
            model_filepath=model_filepath,
        )
    finally:
        read_logger.close()
    return hyperparameters, settings


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn")
    # https://stackoverflow.com/questions/46335842/python-multiprocessing-throws-error-with-argparse-and-pyinstaller
    multiprocessing.freeze_support()
    import os
    import argparse
    import shutil
    import platform
    import keyring
    import json
    import webbrowser
    import neptune.exceptions

    default_experiment_directory = os.path.join(os.path.expanduser("~"), "experiments")
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
        model_filepath="",
    )

    if os.path.exists(last_neptune_run_path):
        try:
            last_neptune_run = get_last_neptune_run(last_neptune_run_path)
            hyperparameters, settings = get_last_hyperparameters_and_settings(
                experiment_directory,
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
        except (
            neptune.common.exceptions.NeptuneInvalidApiTokenException,
            neptune.exceptions.MissingFieldException,
        ):
            last_neptune_run = None
    else:
        last_neptune_run = None

    if getattr(sys, "frozen", False):
        working_directory = sys._MEIPASS
    else:
        working_directory = os.getcwd()

    view = renderer.Renderer(
        settings, os.path.join(working_directory, "assets", "icon.png")
    )
    try:
        if not os.path.exists(experiment_directory):
            os.makedirs(experiment_directory)

        initial_open = True
        while view.open and (initial_open or settings.state == renderer.State.NEW):
            while view.open and settings.state != renderer.State.RUNNING:
                settings = view.update()
                time.sleep(1.0 / 30.0)

            model_changed = (
                settings.latent_size != hyperparameters.latent_dims
            )
            if model_changed:
                hyperparameters = hyperparameters._replace(
                    latent_dims=settings.latent_size,
                )

            if settings.learning_rate != hyperparameters.learning_rate:
                hyperparameters = hyperparameters._replace(
                    learning_rate=settings.learning_rate,
                )


            if model_changed or not initial_open:
                last_neptune_run = None
                shutil.rmtree(experiment_directory, ignore_errors=True)
                os.makedirs(experiment_directory)

            initial_open = False

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

                try:
                    write_logger = logger.Logger(
                        settings.neptune_project_name,
                        settings.neptune_api_token,
                        last_neptune_run,
                        flush_period=1.0,
                    )
                    try:
                        url = write_logger.get_url()
                        webbrowser.open_new_tab(url)

                        # Indicates this is a new experiment.
                        if last_neptune_run is None:
                            cache_neptune_run_id(
                                last_neptune_run_path, write_logger.run_id
                            )
                            write_logger.set_values(
                                {
                                    f"hyperparameters/{key}": value
                                    for key, value in hyperparameters._asdict().items()
                                }
                            )
                            write_logger.upload_file("model", settings.model_filepath)

                        mine, theirs = multiprocessing.Pipe()

                        process = multiprocessing.Process(
                            target=background.experiment_process,
                            args=(
                                theirs,
                                hyperparameters,
                                checkpoint_path,
                                cache_directory,
                                backend,
                            ),
                            name="experiment",
                        )
                        try:
                            process.start()
                            old_settings = settings
                            while view.open and not mine.closed and process.is_alive():
                                new_settings = view.update()

                                if old_settings != new_settings:
                                    settings = old_settings = new_settings
                                    mine.send(settings)

                                while mine.poll(0.0):
                                    message = mine.recv()
                                    if message == "ready":
                                        mine.send(settings)
                                    else:
                                        (
                                            step,
                                            timestamp,
                                            log_values,
                                            log_images,
                                        ) = message
                                        write_logger.append_values(
                                            log_values, step, timestamp
                                        )
                                        write_logger.append_images(
                                            log_images, step, timestamp
                                        )
                        except KeyboardInterrupt:
                            print("Stopping...", flush=True)
                        finally:
                            print("Closing connection!", flush=True)
                            mine.send("kill")
                            mine.close()
                            while process.is_alive():
                                print("Joining...", flush=True)
                                process.join(timeout=1.0)
                            print("Join successful!", flush=True)
                    finally:
                        print("Closing logger!", flush=True)
                        write_logger.close()
                except neptune.common.exceptions.NeptuneInvalidApiTokenException:
                    view.show_error("Invalid Neptune API token.")
                except background.BadUpdateError as error:
                    (message,) = error.args
                    view.show_error(
                        f"Bad training update: {message}. Please restart and reset your experiment."
                    )
                except Exception as exception:
                    raise exception
    finally:
        view.close()
