if __name__ == "__main__":
    import os
    import time
    import neptune
    import numpy as np
    import argparse
    import PIL.Image
    import shutil
    import platform
    import experiment

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
        hyperparameters = experiment.Hyperparameters(latent_dims=16, learning_rate=1e-3)
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
                hyperparameters = experiment.Hyperparameters(
                    latent_dims=run["hyperparameters/latent_dims"].fetch(),
                    learning_rate=run["hyperparameters/learning_rate"].fetch(),
                )
            finally:
                run.stop()
        else:
            raise RuntimeError("No last Neptune run found!")

    this_experiment = experiment.Experiment(
        hyperparameters, checkpoint_path, cache_directory, backend
    )
    try:
        if this_experiment.checkpoint_exists():
            this_experiment.restore()
        else:
            last_neptune_run = None
            this_experiment.reset(0)

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

            for key, value in hyperparameters._asdict().items():
                run[f"hyperparameters/{key}"] = value

            profile = {}
            while True:
                profile.clear()
                start_train_step = time.monotonic()
                metrics = this_experiment.train_step()
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
                    predicted_images = this_experiment.predict(0, image_ids)
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
        this_experiment.close()
