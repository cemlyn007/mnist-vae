import http.client
import os
import jax.numpy as jnp
import jax
import gzip


class Dataset:
    # http://yann.lecun.com/exdb/mnist/
    HOST_URL = "yann.lecun.com"
    TRAIN_IMAGES_RELATIVE_URL = "/exdb/mnist/train-images-idx3-ubyte.gz"
    TRAIN_LABELS_RELATIVE_URL = "/exdb/mnist/train-labels-idx1-ubyte.gz"
    TEST_IMAGES_RELATIVE_URL = "/exdb/mnist/t10k-images-idx3-ubyte.gz"
    TEST_LABELS_RELATIVE_URL = "/exdb/mnist/t10k-labels-idx1-ubyte.gz"

    TRAIN_IMAGES_FILE_NAME = "train-images-idx3-ubyte.gz"
    TRAIN_LABELS_FILE_NAME = "train-labels-idx1-ubyte.gz"
    TEST_IMAGES_FILE_NAME = "t10k-images-idx3-ubyte.gz"
    TEST_LABELS_FILE_NAME = "t10k-labels-idx1-ubyte.gz"

    def __init__(self, cache_directory: str) -> None:
        self._cache_directory = cache_directory

    def download(self) -> None:
        if not os.path.exists(self._cache_directory):
            os.makedirs(self._cache_directory)

        for relative_url, file_name in [
            (Dataset.TRAIN_IMAGES_RELATIVE_URL, Dataset.TRAIN_IMAGES_FILE_NAME),
            (Dataset.TRAIN_LABELS_RELATIVE_URL, Dataset.TRAIN_LABELS_FILE_NAME),
            (Dataset.TEST_IMAGES_RELATIVE_URL, Dataset.TEST_IMAGES_FILE_NAME),
            (Dataset.TEST_LABELS_RELATIVE_URL, Dataset.TEST_LABELS_FILE_NAME),
        ]:
            file_path = os.path.join(self._cache_directory, file_name)
            if not os.path.exists(file_path):
                self._download_file(relative_url, file_path)

    def _download_file(self, relative_url: str, file_path: str) -> None:
        connection = http.client.HTTPConnection(Dataset.HOST_URL)
        try:
            connection.request("GET", relative_url)
            response = connection.getresponse()
            if response.status != http.HTTPStatus.OK:
                raise RuntimeError(
                    f"Failed to download {relative_url} with status {response.status}"
                )
            # else...
            with open(file_path, "wb") as file:
                file.write(response.read())
        finally:
            connection.close()

    def load_train_images(self, device: jax.Device) -> jax.Array:
        return self._load_images(
            os.path.join(self._cache_directory, Dataset.TRAIN_IMAGES_FILE_NAME),
            2051,
            device,
        )

    def load_test_images(self, device: jax.Device) -> jax.Array:
        return self._load_images(
            os.path.join(self._cache_directory, Dataset.TEST_IMAGES_FILE_NAME),
            2051,
            device,
        )

    def _load_images(
        self, file_path: str, expected_magic_number: int, device: jax.Device
    ) -> jax.Array:
        with gzip.open(file_path) as f:
            magic_number = int.from_bytes(f.read(4), byteorder="big", signed=True)
            if magic_number != expected_magic_number:
                raise AssertionError(
                    f"Invalid magic number {magic_number} for MNIST images"
                )
            # else...
            number_of_images = int.from_bytes(f.read(4), byteorder="big", signed=True)
            number_of_rows = int.from_bytes(f.read(4), byteorder="big", signed=True)
            number_of_columns = int.from_bytes(f.read(4), byteorder="big", signed=True)
            images_in_bytes = f.read(
                number_of_images * number_of_rows * number_of_columns
            )

        with jax.default_device(device):
            flat_images = jnp.frombuffer(images_in_bytes, dtype=jnp.uint8)

        # JAX-Metal does not support reshaping on the GPU, so we fallback onto CPU and later
        # will put it back on the GPU.
        if device.platform == "METAL":
            device_for_reshape = jax.devices("cpu")[0]
        else:
            device_for_reshape = device
        images = jax.device_put(flat_images, device_for_reshape).reshape(
            number_of_images, number_of_rows, number_of_columns
        )
        return jax.device_put(images, device)

    def load_train_labels(self, device: jax.Device) -> jax.Array:
        return self._load_labels(
            os.path.join(self._cache_directory, Dataset.TRAIN_LABELS_FILE_NAME),
            2049,
            device,
        )

    def load_test_labels(self, device: jax.Device) -> jax.Array:
        return self._load_labels(
            os.path.join(self._cache_directory, Dataset.TEST_LABELS_FILE_NAME),
            2049,
            device,
        )

    def _load_labels(
        self, file_path: str, expected_magic_number: int, device: jax.Device
    ) -> jax.Array:
        with gzip.open(file_path) as f:
            magic_number = int.from_bytes(f.read(4), byteorder="big", signed=True)
            if magic_number != expected_magic_number:
                raise AssertionError(
                    f"Invalid magic number {magic_number} for MNIST labels"
                )
            # else...
            number_of_labels = int.from_bytes(f.read(4), byteorder="big", signed=True)
            labels_in_bytes = f.read(number_of_labels)
        with jax.default_device(device):
            labels = jnp.frombuffer(labels_in_bytes, dtype=jnp.uint8)
        return labels
