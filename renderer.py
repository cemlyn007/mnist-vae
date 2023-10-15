import tkinter as tk
from tkinter import ttk
import typing
import sys
import enum


class State(enum.Enum):
    NEW = 0
    RUNNING = 1
    PAUSED = 2


class Settings(typing.NamedTuple):
    latent_size: int
    learning_rate: float
    beta: float
    batch_size: int
    predict_interval: int
    tsne_interval: int
    tsne_perplexity: float
    tsne_iterations: int
    checkpoint_interval: int
    checkpoint_max_to_keep: int
    neptune_project_name: str
    neptune_api_token: str
    state: State


class Renderer:
    def __init__(self, settings: Settings, icon_file_path: str) -> None:
        self._settings = settings
        self._root = tk.Tk(className=" MNIST VAE Settings")
        self._root.iconphoto(False, tk.PhotoImage(file=icon_file_path))

        self._add_widgets()
        self._make_dynamic(self._root)

        self._root.protocol("WM_DELETE_WINDOW", self._set_window_closed)

        self.open = True

    def update(self) -> Settings:
        self._root.update()
        return self._settings

    def close(self) -> None:
        self._root.quit()

    def _add_widgets(self) -> None:
        tk.Label(self._root, text="Credentials", anchor="w").grid(columnspan=2, row=0)

        neptune_project_label = tk.Label(
            self._root, text="Neptune Project:", anchor="w"
        )
        neptune_project_label.grid(
            column=0,
            row=1,
            sticky="ew",
        )

        self._neptune_project_text = tk.StringVar(
            self._root, value=str(self._settings.neptune_project_name)
        )
        self._neptune_project_input = tk.Entry(
            self._root, textvariable=self._neptune_project_text
        )
        self._neptune_project_input.grid(
            column=1,
            row=1,
            sticky="ew",
        )

        neptune_api_token_label = tk.Label(
            self._root, text="Neptune API Token:", anchor="w"
        )
        neptune_api_token_label.grid(
            column=0,
            row=2,
            sticky="ew",
        )

        self._neptune_api_token_text = tk.StringVar(
            self._root, value=str(self._settings.neptune_api_token)
        )
        self._neptune_api_token_input = tk.Entry(
            self._root,
            textvariable=self._neptune_api_token_text,
            show="*",
        )
        self._neptune_api_token_input.grid(
            column=1,
            row=2,
            sticky="ew",
        )
        ttk.Separator(self._root, orient="horizontal").grid(
            columnspan=2, row=3, sticky="ew"
        )

        tk.Label(self._root, text="Hyperparameters", anchor="w").grid(
            columnspan=2,
            row=4,
        )

        tk.Label(self._root, text="Latent Size:", anchor="w").grid(
            column=0, row=5, sticky="ew"
        )

        self._latent_size_text = tk.StringVar(
            self._root, value=str(self._settings.latent_size)
        )
        self._latent_size_input = tk.Spinbox(
            self._root,
            textvariable=self._latent_size_text,
            from_=1,
            to=sys.maxsize,
            increment=1,
            validate="focusout",
            validatecommand=self._validate_latent_size,
            command=self._latent_size_callback,
            state="disabled" if self._settings.state != State.NEW else "normal",
        )
        self._latent_size_text.set(str(self._settings.latent_size))
        self._latent_size_input.grid(row=5, column=1, sticky="ew")
        self._latent_size_input.bind("<FocusOut>", self._latent_size_callback)

        tk.Label(self._root, text="Learning Rate:", anchor="w").grid(
            column=0, row=6, sticky="ew"
        )

        self._learning_rate_text = tk.StringVar(
            self._root, value=str(self._settings.learning_rate)
        )
        self._learning_rate_input = tk.Spinbox(
            self._root,
            textvariable=self._learning_rate_text,
            from_=0.0,
            to=100.0,
            increment=0.1,
            validate="focusout",
            validatecommand=self._validate_learning_rate,
            command=self._learning_rate_callback,
        )
        self._learning_rate_text.set(str(self._settings.learning_rate))
        self._learning_rate_input.grid(column=1, row=6, sticky="ew")
        self._learning_rate_input.bind("<FocusOut>", self._learning_rate_callback)

        tk.Label(self._root, text="Beta:", anchor="w").grid(
            column=0, row=7, sticky="ew"
        )
        self._beta_text = tk.StringVar(self._root, value=str(self._settings.beta))
        self._beta_input = tk.Spinbox(
            self._root,
            textvariable=self._beta_text,
            from_=0.0,
            to=100.0,
            increment=0.1,
            validate="focusout",
            validatecommand=self._validate_beta,
            command=self._beta_callback,
        )
        self._beta_text.set(str(self._settings.beta))
        self._beta_input.grid(column=1, row=7, sticky="ew")
        self._beta_input.bind("<FocusOut>", self._beta_callback)

        tk.Label(self._root, text="Batch Size:", anchor="w").grid(
            column=0, row=8, sticky="ew"
        )
        self._batch_size_text = tk.StringVar(
            self._root, value=str(self._settings.batch_size)
        )
        self._batch_size_input = tk.Spinbox(
            self._root,
            textvariable=self._batch_size_text,
            from_=0,
            to=sys.maxsize,
            increment=1,
            validate="focusout",
            validatecommand=self._validate_batch_size,
            command=self._batch_size_callback,
        )
        self._batch_size_text.set(str(self._settings.batch_size))
        self._batch_size_input.grid(column=1, row=8, sticky="ew")
        self._batch_size_input.bind("<FocusOut>", self._batch_size_callback)

        ttk.Separator(self._root, orient="horizontal").grid(
            columnspan=2, row=9, sticky="ew"
        )

        tk.Label(self._root, text="Monitor", anchor="w").grid(columnspan=2, row=10)

        tk.Label(self._root, text="Predict Interval:", anchor="w").grid(
            column=0, row=11, sticky="ew"
        )

        self._predict_interval_text = tk.StringVar(
            self._root, value=str(self._settings.predict_interval)
        )
        self._predict_interval_input = tk.Spinbox(
            self._root,
            textvariable=self._predict_interval_text,
            from_=0,
            to_=sys.maxsize,
            increment=1,
            validate="focusout",
            validatecommand=self._validate_predict_interval,
            command=self._predict_interval_callback,
        )
        self._predict_interval_text.set(str(self._settings.predict_interval))
        self._predict_interval_input.grid(column=1, row=11, sticky="ew")
        self._predict_interval_input.bind("<FocusOut>", self._predict_interval_callback)

        tk.Label(self._root, text="t-SNE Interval:", anchor="w").grid(
            column=0, row=12, sticky="ew"
        )

        self._tsne_interval_text = tk.StringVar(
            self._root, value=str(self._settings.tsne_interval)
        )
        self._tsne_interval_input = tk.Spinbox(
            self._root,
            textvariable=self._tsne_interval_text,
            from_=0,
            to_=sys.maxsize,
            increment=1,
            validate="focusout",
            validatecommand=self._validate_tsne_interval,
            command=self._tsne_interval_callback,
        )
        self._tsne_interval_text.set(str(self._settings.tsne_interval))
        self._tsne_interval_input.grid(column=1, row=12, sticky="ew")
        self._tsne_interval_input.bind("<FocusOut>", self._tsne_interval_callback)

        tk.Label(self._root, text="t-SNE Perplexity:", anchor="w").grid(
            column=0, row=13, sticky="ew"
        )

        self._tsne_perplexity_text = tk.StringVar(
            self._root, value=str(self._settings.tsne_perplexity)
        )
        self._tsne_perplexity_input = tk.Spinbox(
            self._root,
            textvariable=self._tsne_perplexity_text,
            from_=0,
            to_=sys.maxsize,
            increment=1,
            validate="focusout",
            validatecommand=self._validate_tsne_perplexity,
            command=self._tsne_perplexity_callback,
        )
        self._tsne_perplexity_text.set(str(self._settings.tsne_perplexity))
        self._tsne_perplexity_input.grid(column=1, row=13, sticky="ew")
        self._tsne_perplexity_input.bind("<FocusOut>", self._tsne_perplexity_callback)

        tk.Label(self._root, text="t-SNE Iterations:", anchor="w").grid(
            column=0, row=14, sticky="ew"
        )

        self._tsne_iterations_text = tk.StringVar(
            self._root, value=str(self._settings.tsne_iterations)
        )
        self._tsne_iterations_input = tk.Spinbox(
            self._root,
            textvariable=self._tsne_iterations_text,
            from_=1,
            to_=sys.maxsize,
            increment=50,
            validate="focusout",
            validatecommand=self._validate_tsne_iterations,
            command=self._tsne_iterations_callback,
        )
        self._tsne_iterations_text.set(str(self._settings.tsne_iterations))
        self._tsne_iterations_input.grid(column=1, row=14, sticky="ew")
        self._tsne_iterations_input.bind("<FocusOut>", self._tsne_iterations_callback)

        tk.Label(self._root, text="Checkpoint Interval:", anchor="w").grid(
            column=0, row=15, sticky="ew"
        )

        self._checkpoint_interval_text = tk.StringVar(
            self._root, value=str(self._settings.checkpoint_interval)
        )
        self._checkpoint_interval_input = tk.Spinbox(
            self._root,
            textvariable=self._checkpoint_interval_text,
            from_=0,
            to_=sys.maxsize,
            increment=1,
            validate="focusout",
            validatecommand=self._validate_checkpoint_interval,
            command=self._checkpoint_interval_callback,
        )
        self._checkpoint_interval_text.set(str(self._settings.checkpoint_interval))
        self._checkpoint_interval_input.grid(column=1, row=15, sticky="ew")
        self._checkpoint_interval_input.bind(
            "<FocusOut>", self._checkpoint_interval_callback
        )

        tk.Label(self._root, text="Checkpoint Max To Keep:", anchor="w").grid(
            column=0, row=16, sticky="ew"
        )

        self._checkpoint_max_to_keep_text = tk.StringVar(
            self._root, value=str(self._settings.checkpoint_max_to_keep)
        )
        self._checkpoint_max_to_keep_input = tk.Spinbox(
            self._root,
            textvariable=self._checkpoint_max_to_keep_text,
            from_=0,
            to_=sys.maxsize,
            increment=1,
            validate="focusout",
            validatecommand=self._validate_checkpoint_max_to_keep,
            command=self._checkpoint_max_to_keep_callback,
        )
        self._checkpoint_max_to_keep_text.set(
            str(self._settings.checkpoint_max_to_keep)
        )
        self._checkpoint_max_to_keep_input.grid(column=1, row=16, sticky="ew")
        self._checkpoint_max_to_keep_input.bind(
            "<FocusOut>", self._checkpoint_max_to_keep_callback
        )

        ttk.Separator(self._root, orient="horizontal").grid(
            row=19, columnspan=2, sticky="ew"
        )

        self._state_button = tk.Button(
            self._root, text="Start", command=self._start_callback
        )
        self._state_button.grid(columnspan=2, row=17, sticky="ew")

    def _latent_size_callback(self, event=None) -> None:
        value = self._latent_size_text.get().strip()
        self._settings = self._settings._replace(latent_size=int(value))

    def _validate_latent_size(self) -> bool:
        value = self._latent_size_text.get().strip()
        try:
            int(value)
            return True
        except ValueError:
            return False

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

    def _batch_size_callback(self, event=None) -> None:
        value = self._batch_size_text.get().strip()
        self._settings = self._settings._replace(batch_size=int(value))

    def _validate_batch_size(self) -> bool:
        value = self._batch_size_text.get().strip()
        try:
            int(value)
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

    def _validate_predict_interval(self) -> bool:
        value = self._predict_interval_text.get().strip()
        try:
            int(value)
            return True
        except ValueError:
            return False

    def _tsne_interval_callback(self, event=None) -> None:
        value = self._tsne_interval_text.get().strip()
        self._settings = self._settings._replace(tsne_interval=int(value))

    def _validate_tsne_interval(self) -> bool:
        value = self._tsne_interval_text.get().strip()
        try:
            int(value)
            return True
        except ValueError:
            return False

    def _tsne_perplexity_callback(self, event=None) -> None:
        value = self._tsne_perplexity_text.get().strip()
        self._settings = self._settings._replace(tsne_perplexity=float(value))

    def _validate_tsne_perplexity(self) -> bool:
        value = self._tsne_perplexity_text.get().strip()
        try:
            float(value)
            return True
        except ValueError:
            return False

    def _tsne_iterations_callback(self, event=None) -> None:
        value = self._tsne_iterations_text.get().strip()
        self._settings = self._settings._replace(tsne_iterations=int(value))

    def _validate_tsne_iterations(self) -> bool:
        value = self._tsne_iterations_text.get().strip()
        try:
            int(value)
            return True
        except ValueError:
            return False

    def _checkpoint_interval_callback(self, event=None) -> None:
        value = self._checkpoint_interval_text.get().strip()
        self._settings = self._settings._replace(checkpoint_interval=int(value))

    def _validate_checkpoint_interval(self) -> bool:
        value = self._checkpoint_interval_text.get().strip()
        try:
            int(value)
            return True
        except ValueError:
            return False

    def _checkpoint_max_to_keep_callback(self, event=None) -> None:
        value = self._checkpoint_max_to_keep_text.get().strip()
        self._settings = self._settings._replace(checkpoint_max_to_keep=int(value))

    def _validate_checkpoint_max_to_keep(self) -> bool:
        value = self._checkpoint_max_to_keep_text.get().strip()
        try:
            int(value)
            return True
        except ValueError:
            return False

    def _start_callback(self, event=None) -> None:
        if self._settings.state == State.NEW:
            self._latent_size_input.configure(state="disabled")
            self._state_button.config(text="Pause")
            self._neptune_project_input.configure(state="disabled")
            self._neptune_api_token_input.configure(state="disabled")
            self._settings = self._settings._replace(
                state=State.RUNNING,
                neptune_project_name=self._neptune_project_text.get().strip(),
                neptune_api_token=self._neptune_api_token_text.get().strip(),
            )
        elif self._settings.state == State.PAUSED:
            self._state_button.config(text="Pause")
            self._settings = self._settings._replace(state=State.RUNNING)
        elif self._settings.state == State.RUNNING:
            self._state_button.config(text="Resume")
            self._settings = self._settings._replace(state=State.PAUSED)
        else:
            raise ValueError("Invalid state")

    def _set_window_closed(self) -> None:
        self.open = False

    def _make_dynamic(self, widget: tk.Widget) -> None:
        col_count, row_count = widget.grid_size()

        for i in range(col_count):
            widget.grid_columnconfigure(i, weight=1 if i == 1 else 0)

        for child in widget.children.values():
            child.grid_configure(sticky="ew")
            self._make_dynamic(child)
