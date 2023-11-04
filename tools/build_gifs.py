import neptune
import os
import PIL.Image
import tqdm
import operator
import numpy as np
import neptune.types
import concurrent.futures
import tempfile


class GifRenderer:
    def __init__(self, run_id: str, api_key: str, project: str, directory: str) -> None:
        self.run_id = run_id
        self.api_key = api_key
        self.project = project
        self.directory = directory

    def render(self):
        with tempfile.TemporaryDirectory() as directory:
            data = self._get_data_paths(directory)
            print(f"Rendering GIF for run {self.run_id}...")

            work = [
                lambda: self._create_gif(
                    data["train"]["embeddings"],
                    os.path.join(self.directory, "embeddings.gif"),
                )
            ]
            for digit, digit_directory in data["train"]["predicted_images"].items():
                work.append(
                    lambda: self._create_gif(
                        digit_directory,
                        os.path.join(self.directory, f"{digit}.gif"),
                    )
                )
            work.append(
                lambda: self._merge_digit_gifs(
                    data["train"]["predicted_images"].items(),
                    os.path.join(self.directory, "digits.gif"),
                )
            )
            work.append(
                lambda: self._merge_embeddings_with_digits(
                    data["train"]["embeddings"],
                    data["train"]["predicted_images"].items(),
                    os.path.join(self.directory, "ai.gif"),
                )
            )

            if not os.path.exists(self.directory):
                os.makedirs(self.directory)

            for make_gif in tqdm.tqdm(work, total=len(work)):
                make_gif()

    def _create_gif(self, source_directory: str, destination: str) -> None:
        filenames = os.listdir(source_directory)
        filenames.sort(key=lambda filename: int(filename.split(".")[0]))
        images = [
            PIL.Image.open(os.path.join(source_directory, filename))
            for filename in filenames
        ]
        frame = images[0].copy()
        frame.save(
            destination,
            format="GIF",
            append_images=images[1:],
            save_all=True,
            duration=100,
            loop=0,
            optimize=True,
        )

    def _merge_digit_gifs(
        self, digit_directories: list[tuple[int, str]], destination: str
    ) -> None:
        digits_images = sorted(
            (
                (
                    digit,
                    [
                        np.array(
                            PIL.Image.open(os.path.join(digit_directory, filename))
                        )
                        for filename in sorted(
                            os.listdir(digit_directory),
                            key=lambda filename: int(filename.split(".")[0]),
                        )
                    ],
                )
                for digit, digit_directory in digit_directories
            ),
            key=operator.itemgetter(0),
        )

        n_frames = {len(images) for _, images in digits_images}
        if len(n_frames) != 1:
            print(
                f"Expected all digits images to have the same number of images, but got {n_frames}."
            )
        # else...
        for _, images in digits_images:
            for image in images:
                if (
                    image.shape[0] != images[0].shape[0]
                    or image.shape[1] != images[0].shape[1]
                ):
                    raise ValueError(
                        f"Expected all digits images to have the same dimensions, but got {images}."
                    )
        n_frame = min(n_frames)
        images = [
            PIL.Image.fromarray(
                np.hstack(
                    [
                        np.array(digit_images[frame_index])
                        for (digit, digit_images) in digits_images
                    ]
                )
            )
            for frame_index in range(n_frame)
        ]
        frame = images[0].copy()
        frame.save(
            destination,
            format="GIF",
            append_images=images[1:],
            save_all=True,
            duration=100,
            loop=0,
        )

    def _merge_embeddings_with_digits(
        self,
        embeddings_directory: str,
        digit_directories: list[tuple[int, str]],
        destination: str,
    ) -> None:
        embedding_images = [
            np.array(PIL.Image.open(os.path.join(embeddings_directory, filename)))
            for filename in sorted(
                os.listdir(embeddings_directory),
                key=lambda filename: int(filename.split(".")[0]),
            )
        ]

        digits_images = sorted(
            (
                (
                    digit,
                    [
                        np.array(
                            PIL.Image.open(os.path.join(digit_directory, filename))
                        )
                        for filename in sorted(
                            os.listdir(digit_directory),
                            key=lambda filename: int(filename.split(".")[0]),
                        )
                    ],
                )
                for digit, digit_directory in digit_directories
            ),
            key=operator.itemgetter(0),
        )
        n_frames = {len(images) for _, images in digits_images}
        if len(n_frames) != 1:
            print(
                f"Expected all digits images to have the same number of images, but got {n_frames}."
            )
        # else...
        for _, images in digits_images:
            for image in images:
                if (
                    image.shape[0] != images[0].shape[0]
                    or image.shape[1] != images[0].shape[1]
                ):
                    raise ValueError(
                        f"Expected all digits images to have the same dimensions, but got {images}."
                    )
        n_frames = min(min(n_frames), len(embedding_images))
        stacked_digits_images = [
            np.hstack(
                [
                    np.array(digit_images[frame_index])
                    for (digit, digit_images) in digits_images
                ]
            )
            for frame_index in range(n_frames)
        ]

        frames = []
        for frame_index in range(n_frames):
            image = PIL.Image.fromarray(stacked_digits_images[frame_index])
            frames.append(
                PIL.Image.fromarray(
                    np.vstack(
                        [
                            np.array(
                                image.resize(
                                    (1200, round((1200 / 280) * image.size[1]))
                                ).convert("RGBA")
                            ),
                            embedding_images[frame_index],
                        ]
                    )
                )
            )
        frame = frames[0].copy()
        frame.save(
            destination,
            format="GIF",
            append_images=frames[1:],
            save_all=True,
            duration=100,
            loop=0,
        )

    def _get_data_paths(self, directory: str) -> dict[str, any]:
        data_paths = {}
        run = neptune.init_run(
            with_id=self.run_id,
            project=self.project,
            api_token=self.api_key,
            mode="read-only",
            capture_traceback=False,
        )
        try:
            work = []
            structure = run.get_structure()
            if "train" not in structure:
                raise ValueError("No train images found in the run.")
            # else...
            train_structure = structure["train"]
            data_paths["train"] = {}
            embeddings_directory = os.path.join(directory, "train", "embeddings")
            data_paths["train"]["embeddings"] = embeddings_directory
            work.append((train_structure["embeddings"], embeddings_directory))
            if "predicted_images" not in train_structure:
                raise ValueError("No predicted images found in the run.")
            # else...
            predicted_images_structure = train_structure["predicted_images"]
            predicted_images_directory = os.path.join(
                directory, "train", "predicted_images"
            )
            data_paths["train"]["predicted_images"] = {}
            for digit, digit_structure in predicted_images_structure.items():
                data_paths["train"]["predicted_images"][digit] = os.path.join(
                    predicted_images_directory, digit
                )
                work.append(
                    (digit_structure, data_paths["train"]["predicted_images"][digit])
                )

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        lambda file_series, directory: file_series.download(directory),
                        file_series,
                        directory,
                    )
                    for file_series, directory in work
                ]
                for future in tqdm.tqdm(
                    concurrent.futures.as_completed(futures), total=len(futures)
                ):
                    future.result()
        finally:
            run.stop()
        return data_paths


if __name__ == "__main__":
    import argparse

    argument_parser = argparse.ArgumentParser(
        description="Build a GIFs from a Neptune Run."
    )
    argument_parser.add_argument(
        "--run", type=str, help="Run ID", default=None, required=False
    )
    argument_parser.add_argument(
        "--path", type=str, help="Path to save the GIF", default=None
    )
    argument_parser.add_argument(
        "--api-key", type=str, help="Neptune API key", default=None
    )
    argument_parser.add_argument("--project", type=str, help="Neptune project")

    arguments = argument_parser.parse_args()

    directory = arguments.path

    if arguments.api_key is None:
        api_key = os.environ["NEPTUNE_API_TOKEN"]
    else:
        api_key = arguments.api_key

    if arguments.run is None:
        project = neptune.init_project(project=arguments.project, mode="read-only")
        try:
            run_id = (
                project.fetch_runs_table().to_rows()[0].get_attribute_value("sys/id")
            )
        finally:
            project.stop()
        directory = os.path.join(directory, run_id)
    else:
        run_id = arguments.run

    GifRenderer(
        run_id=run_id,
        api_key=api_key,
        project=arguments.project,
        directory=directory,
    ).render()
