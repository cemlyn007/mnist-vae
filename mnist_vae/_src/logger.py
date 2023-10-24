import neptune
import PIL.Image


class Logger:
    def __init__(
        self,
        neptune_project: str,
        neptune_api_token: str,
        run_id: str | None = None,
        flush_period: float = 1.0,
        read_only: bool = False,
    ) -> None:
        self._run = neptune.init_run(
            with_id=run_id,
            project=neptune_project,
            api_token=neptune_api_token,
            capture_hardware_metrics=False,
            capture_stderr=False,
            capture_stdout=False,
            capture_traceback=False,
            flush_period=flush_period,
            mode="read-only" if read_only else "async",
            git_ref=False,
        )

    def get_url(self) -> str:
        return self._run.get_url()

    def get_int(self, key: str) -> int:
        return self._run[key].fetch()

    def get_float(self, key: str) -> float:
        return self._run[key].fetch()

    def get_last_int(self, key: str) -> int:
        value = self._run[key].fetch_last()
        if not value.is_integer():
            raise ValueError(f"Value {value} is not an integer")
        # else...
        return int(value)

    def get_last_float(self, key: str) -> float:
        return self._run[key].fetch_last()

    def set_values(self, values: dict[str, any]) -> None:
        for key, value in values.items():
            self._run[key] = value

    def append_values(
        self, metrics: dict[str, float], step: int, timestamp: float
    ) -> None:
        for key, value in metrics.items():
            self._run[key].append(value, timestamp=timestamp, step=step)

    def append_images(
        self, images: dict[str, PIL.Image.Image], step: int, timestamp: float
    ) -> None:
        for key, image in images.items():
            self._run[key].append(
                image,
                timestamp=timestamp,
                step=step,
            )

    def upload_file(self, name: str, path: str) -> None:
        self._run[name].upload(path)

    def download_file(self, name: str, destination: str) -> None:
        self._run[name].download(destination)

    @property
    def run_id(self) -> str:
        return self._run["sys/id"].fetch()

    def close(self) -> None:
        self._run.stop()
