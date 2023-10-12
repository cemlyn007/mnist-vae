import tkinter as tk
import typing
import sys


class Settings(typing.NamedTuple):
    learning_rate: float
    beta: float
    predict_interval: int


class Renderer:
    def __init__(self, settings: Settings, icon_file_path: str) -> None:
        self._settings = settings
        self._root = tk.Tk(className=" MNIST VAE Settings")

        self._root.iconphoto(False, tk.PhotoImage(file=icon_file_path))
        tk.Frame(self._root)
        self._add_hyperparameter_frame(0)
        self._add_monitor_frame(1)

        self._root.protocol("WM_DELETE_WINDOW", self._set_window_closed)

        self.open = True

    def update(self) -> Settings:
        self._root.update()
        return self._settings

    def close(self) -> None:
        self._root.quit()

    def _add_hyperparameter_frame(self, row: int) -> None:
        hyperparameter_frame = tk.LabelFrame(self._root, text="Hyperparameters")
        hyperparameter_frame.grid(row=row)

        tk.Label(hyperparameter_frame, text="Learning Rate:", padx=5, pady=5).grid(
            column=0, row=0
        )

        self._learning_rate_text = tk.StringVar(
            hyperparameter_frame, value=str(self._settings.learning_rate)
        )
        self._learning_rate_input = tk.Spinbox(
            hyperparameter_frame,
            textvariable=self._learning_rate_text,
            from_=0.0,
            to=100.0,
            increment=0.1,
            validate="focusout",
            validatecommand=self._validate_learning_rate,
            command=self._learning_rate_callback,
        )
        self._learning_rate_text.set(str(self._settings.learning_rate))
        self._learning_rate_input.grid(column=1, row=0)
        self._learning_rate_input.bind("<FocusOut>", self._learning_rate_callback)

        tk.Label(hyperparameter_frame, text="Beta:", padx=5, pady=5).grid(
            column=0, row=1
        )
        self._beta_text = tk.StringVar(
            hyperparameter_frame, value=str(self._settings.beta)
        )
        self._beta_input = tk.Spinbox(
            hyperparameter_frame,
            textvariable=self._beta_text,
            from_=0.0,
            to=100.0,
            increment=0.1,
            validate="focusout",
            validatecommand=self._validate_beta,
            command=self._beta_callback,
        )
        self._beta_text.set(str(self._settings.beta))
        self._beta_input.grid(column=1, row=1)
        self._beta_input.bind("<FocusOut>", self._beta_callback)

    def _add_monitor_frame(self, row: int) -> None:
        frame = tk.LabelFrame(self._root, text="Monitor")
        frame.grid(row=row)

        tk.Label(frame, text="Predict Interval:").grid(column=0, row=0)

        self._predict_interval_text = tk.StringVar(
            frame, value=str(self._settings.predict_interval)
        )
        self._predict_interval_input = tk.Spinbox(
            frame,
            textvariable=self._predict_interval_text,
            from_=0,
            to_=sys.maxsize,
            increment=1,
            validate="focusout",
            validatecommand=self._validate_predict_interval,
            command=self._predict_interval_callback,
        )
        self._predict_interval_text.set(str(self._settings.predict_interval))
        self._predict_interval_input.grid(column=1, row=0)
        self._predict_interval_input.bind("<FocusOut>", self._predict_interval_callback)

    def _beta_callback(self, event=None) -> None:
        value = self._beta_text.get().strip()
        self._settings = self._settings._replace(beta=float(value))

    def _validate_beta(self) -> bool:
        value = self._beta_text.get().strip()
        try:
            float(value)
            return True
        except ValueError:
            return False

    def _learning_rate_callback(self, event=None) -> None:
        value = self._learning_rate_text.get().strip()
        self._settings = self._settings._replace(learning_rate=float(value))

    def _validate_learning_rate(self) -> bool:
        value = self._learning_rate_text.get().strip()
        try:
            float(value)
            return True
        except ValueError:
            return False

    def _predict_interval_callback(self, event=None) -> None:
        value = self._predict_interval_text.get().strip()
        self._settings = self._settings._replace(predict_interval=int(value))
        print("New predict interval:", self._settings.predict_interval)

    def _validate_predict_interval(self) -> bool:
        value = self._predict_interval_text.get().strip()
        try:
            int(value)
            return True
        except ValueError:
            return False

    def _set_window_closed(self) -> None:
        self.open = False
