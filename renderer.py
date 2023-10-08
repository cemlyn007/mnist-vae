import tkinter as tk
import typing


class Settings(typing.NamedTuple):
    beta: float


class Renderer:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._root = tk.Tk(className=" MNIST VAE Settings")
        tk.Frame(self._root)

        tk.Label(self._root, text="Beta:", padx=5, pady=5).grid(column=0, row=0)
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
        self._beta_input.grid(column=1, row=0)
        self._beta_input.bind("<FocusOut>", self._beta_callback)
        self._root.protocol("WM_DELETE_WINDOW", self._set_window_closed)

        self.open = True

    def update(self):
        self._root.update()
        return self._settings

    def close(self):
        self._root.quit()

    def _beta_callback(self):
        value = self._beta_text.get()
        self._settings = self._settings._replace(beta=float(value))

    def _validate_beta(self):
        value = self._beta_text.get().strip()
        try:
            float(value)
            return True
        except ValueError:
            return False

    def _set_window_closed(self):
        self.open = False
