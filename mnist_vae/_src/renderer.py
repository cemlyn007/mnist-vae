import tkinter as tk
from tkinter import ttk
import typing
import sys
import enum
from tkinter import messagebox
import tkinter.filedialog


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
    model_filepath: str
    state: State


class Renderer:
    def __init__(self, settings: Settings, icon_file_path: str) -> None:
        self._settings = settings
        self._root = tk.Tk(className=" MNIST VAE Settings")

        self._root.attributes("-alpha", 0)

        self._root.iconphoto(False, tk.PhotoImage(file=icon_file_path))

        frame = tk.Frame(self._root)
        frame.pack(fill="both", expand=True)

        canvas = tk.Canvas(frame)
        scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        canvas_frame = tk.Frame(canvas)

        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        self._add_widgets(canvas_frame)

        self._make_dynamic(canvas_frame)
        canvas_frame.pack(fill="both", expand=True, anchor="nw")

        self._root.update_idletasks()
        canvas.create_window(
            (0, 0),
            window=canvas_frame,
            anchor="nw",
        )

        self._root.minsize(self._root.winfo_reqwidth(), 0)
        self._root.maxsize(self._root.winfo_reqwidth(), self._root.winfo_reqheight())

        self._root.attributes("-alpha", 1)
        self._root.update_idletasks()

        self._root.protocol("WM_DELETE_WINDOW", self._set_window_closed)

        self.open = True

    def update(self) -> Settings:
        self._root.update()
        return self._settings

    def close(self) -> None:
        self._root.quit()

    def _add_widgets(self, frame: tk.Frame) -> None:
        tk.Label(frame, text="Credentials", anchor="w").grid(columnspan=2, row=0)

        neptune_project_label = tk.Label(frame, text="Neptune Project:", anchor="w")
        neptune_project_label.grid(
            column=0,
            row=1,
            sticky="ew",
        )

        self._neptune_project_text = tk.StringVar(
            frame, value=str(self._settings.neptune_project_name)
        )
        self._neptune_project_input = tk.Entry(
            frame,
            textvariable=self._neptune_project_text,
            state="disabled" if self._settings.state != State.NEW else "normal",
            validate="focusout",
            validatecommand=self._validate_neptune_project,
        )
        self._neptune_project_input.grid(
            column=1,
            row=1,
            sticky="ew",
        )

        neptune_api_token_label = tk.Label(
            frame,
            text="Neptune API Token:",
            anchor="w",
        )
        neptune_api_token_label.grid(
            column=0,
            row=2,
            sticky="ew",
        )

        self._neptune_api_token_text = tk.StringVar(
            frame, value=str(self._settings.neptune_api_token)
        )
        self._neptune_api_token_input = tk.Entry(
            frame,
            textvariable=self._neptune_api_token_text,
            show="*",
            validate="focusout",
            validatecommand=self._validate_neptune_api_token,
        )
        self._neptune_api_token_input.grid(
            column=1,
            row=2,
            sticky="ew",
        )
        ttk.Separator(frame, orient="horizontal").grid(columnspan=2, row=3, sticky="ew")

        tk.Label(frame, text="Hyperparameters", anchor="w").grid(
            columnspan=2,
            row=4,
        )

        tk.Label(frame, text="Latent Size:", anchor="w").grid(
            column=0, row=5, sticky="ew"
        )

        self._latent_size_text = tk.StringVar(
            frame, value=str(self._settings.latent_size)
        )
        self._latent_size_input = tk.Spinbox(
            frame,
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

        tk.Label(frame, text="Learning Rate:", anchor="w").grid(
            column=0, row=6, sticky="ew"
        )

        self._learning_rate_text = tk.StringVar(
            frame, value=str(self._settings.learning_rate)
        )
        self._learning_rate_input = tk.Spinbox(
            frame,
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

        tk.Label(frame, text="Beta:", anchor="w").grid(column=0, row=7, sticky="ew")
        self._beta_text = tk.StringVar(frame, value=str(self._settings.beta))
        self._beta_input = tk.Spinbox(
            frame,
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

        tk.Label(frame, text="Batch Size:", anchor="w").grid(
            column=0, row=8, sticky="ew"
        )
        self._batch_size_text = tk.StringVar(
            frame, value=str(self._settings.batch_size)
        )
        self._batch_size_input = tk.Spinbox(
            frame,
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

        ttk.Separator(frame, orient="horizontal").grid(columnspan=2, row=9, sticky="ew")

        tk.Label(frame, text="Monitor", anchor="w").grid(columnspan=2, row=10)

        tk.Label(frame, text="Predict Interval:", anchor="w").grid(
            column=0, row=11, sticky="ew"
        )

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
        self._predict_interval_input.grid(column=1, row=11, sticky="ew")
        self._predict_interval_input.bind("<FocusOut>", self._predict_interval_callback)

        tk.Label(frame, text="t-SNE Interval:", anchor="w").grid(
            column=0, row=12, sticky="ew"
        )

        self._tsne_interval_text = tk.StringVar(
            frame, value=str(self._settings.tsne_interval)
        )
        self._tsne_interval_input = tk.Spinbox(
            frame,
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

        tk.Label(frame, text="t-SNE Perplexity:", anchor="w").grid(
            column=0, row=13, sticky="ew"
        )

        self._tsne_perplexity_text = tk.StringVar(
            frame, value=str(self._settings.tsne_perplexity)
        )
        self._tsne_perplexity_input = tk.Spinbox(
            frame,
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

        tk.Label(frame, text="t-SNE Iterations:", anchor="w").grid(
            column=0, row=14, sticky="ew"
        )

        self._tsne_iterations_text = tk.StringVar(
            frame, value=str(self._settings.tsne_iterations)
        )
        self._tsne_iterations_input = tk.Spinbox(
            frame,
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

        tk.Label(frame, text="Checkpoint Interval:", anchor="w").grid(
            column=0, row=15, sticky="ew"
        )

        self._checkpoint_interval_text = tk.StringVar(
            frame, value=str(self._settings.checkpoint_interval)
        )
        self._checkpoint_interval_input = tk.Spinbox(
            frame,
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

        tk.Label(frame, text="Checkpoint Max To Keep:", anchor="w").grid(
            column=0, row=16, sticky="ew"
        )

        self._checkpoint_max_to_keep_text = tk.StringVar(
            frame, value=str(self._settings.checkpoint_max_to_keep)
        )
        self._checkpoint_max_to_keep_input = tk.Spinbox(
            frame,
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

        ttk.Separator(frame, orient="horizontal").grid(
            row=17, columnspan=2, sticky="ew"
        )

        self._upload_button = tk.Button(
            frame,
            text="Upload Model",
            command=self._upload_action,
            state="disabled" if self._settings.state != State.NEW else "normal",
        )
        self._upload_button.grid(row=18, columnspan=2, sticky="ew")

        self._state_button = tk.Button(
            frame,
            text="Start" if self._settings.state == State.NEW else "Resume",
            command=self._start_callback,
        )
        self._state_button.grid(column=0, row=19, sticky="ew")

        self._reset_button = tk.Button(
            frame,
            text="Reset",
            command=self._reset_callback,
        )
        self._reset_button.grid(column=1, row=19, sticky="ew")

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

    def _neptune_project_callback(self, event=None) -> None:
        value = self._neptune_project_text.get().strip()
        self._settings = self._settings._replace(neptune_project_name=value)

    def _validate_neptune_project(self) -> bool:
        value = self._neptune_project_text.get().strip()
        components = value.split("/")
        if len(components) != 2:
            return False
        # else...
        owner, project = value.split("/")
        if " " in owner or " " in project:
            return False
        # else...
        return owner != "" and project != ""

    def _neptune_api_token_callback(self, event=None) -> None:
        value = self._neptune_api_token_text.get().strip()
        self._settings = self._settings._replace(neptune_api_token=value)

    def _validate_neptune_api_token(self) -> bool:
        value = self._neptune_api_token_text.get().strip()
        return len(value) > 0

    def _start_callback(self, event=None) -> None:
        if self._settings.state == State.NEW:
            for validate_function, widget_callback in [
                (self._validate_batch_size, self._batch_size_callback),
                (self._validate_beta, self._beta_callback),
                (
                    self._validate_checkpoint_interval,
                    self._checkpoint_interval_callback,
                ),
                (
                    self._validate_checkpoint_max_to_keep,
                    self._checkpoint_max_to_keep_callback,
                ),
                (self._validate_latent_size, self._latent_size_callback),
                (self._validate_learning_rate, self._learning_rate_callback),
                (self._validate_predict_interval, self._predict_interval_callback),
                (self._validate_tsne_interval, self._tsne_interval_callback),
                (self._validate_tsne_iterations, self._tsne_iterations_callback),
                (self._validate_tsne_perplexity, self._tsne_perplexity_callback),
                (self._validate_neptune_project, self._neptune_project_callback),
                (self._validate_neptune_api_token, self._neptune_api_token_callback),
            ]:
                if validate_function():
                    widget_callback()
                else:
                    print("Error: Invalid input", flush=True)
                    self.show_error("Error: Invalid input!")
                    return

            if self._settings.model_filepath == "":
                self.show_error("Error: Please upload a model file!")
                return

            self._latent_size_input.configure(state="disabled")
            self._state_button.config(text="Pause")
            self._neptune_project_input.configure(state="disabled")
            self._neptune_api_token_input.configure(state="disabled")
            self._reset_button.configure(state="disabled")
            self._upload_button.configure(state="disabled")
            self._settings = self._settings._replace(state=State.RUNNING)
        elif self._settings.state == State.PAUSED:
            self._state_button.config(text="Pause")
            self._reset_button.configure(state="disabled")
            self._settings = self._settings._replace(state=State.RUNNING)
        elif self._settings.state == State.RUNNING:
            self._state_button.config(text="Resume")
            self._reset_button.configure(state="normal")
            self._settings = self._settings._replace(state=State.PAUSED)
        else:
            raise ValueError("Invalid state")

    def _reset_callback(self, event=None) -> None:
        self._settings = self._settings._replace(state=State.NEW)
        self._latent_size_input.configure(state="normal")
        self._state_button.config(text="Start")
        self._neptune_project_input.configure(state="normal")
        self._neptune_api_token_input.configure(state="normal")
        self._reset_button.configure(state="normal")
        self._upload_button.configure(state="normal")
        self._settings = self._settings._replace(model_filepath="")

    def _set_window_closed(self) -> None:
        self.open = False

    def _make_dynamic(self, widget: tk.Widget) -> None:
        if isinstance(widget, tk.Scrollbar):
            return
        # else...
        if widget.winfo_manager() != "pack":
            col_count, row_count = widget.grid_size()
            for i in range(col_count):
                widget.grid_columnconfigure(i, weight=1 if i == 1 else 0)

        for child in widget.children.values():
            if child.winfo_manager() != "pack":
                child.grid_configure(sticky="ew")

            self._make_dynamic(child)

    def show_error(self, message: str) -> None:
        messagebox.showerror("Error", message)

    def _upload_action(self, event=None) -> None:
        filename = tkinter.filedialog.askopenfilename()
        self._settings = self._settings._replace(model_filepath=filename)
