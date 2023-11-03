from matplotlib.backends import backend_agg
import matplotlib.figure
import matplotlib.cm
import matplotlib.patches
import jax
import PIL.Image
import numpy as np
from mnist_vae._src.tsne import algorithm

matplotlib.use("Agg")


class Renderer:
    def __init__(self, backend: str) -> None:
        self._image_width = 720
        self._image_height = 720
        self._padding_proportion = 0.25
        self._point_radius = 3
        self._color_map = matplotlib.cm.rainbow(np.linspace(0, 1, 10))
        self._fig = matplotlib.figure.Figure(figsize=(6, 6), dpi=200, layout="tight")
        self._fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        self._ax = self._fig.add_axes((0.0, 0.0, 1.0, 1.0))
        self._ax.legend(
            handles=[
                matplotlib.patches.Patch(color=c, label=str(i))
                for i, c in enumerate(self._color_map)
            ],
            loc="upper right",
            borderaxespad=0.0,
        )
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self._ax.get_xaxis().set_visible(False)
        self._ax.get_yaxis().set_visible(False)
        self._ax.autoscale(enable=True, axis="both")
        self._canvas = backend_agg.FigureCanvasAgg(self._fig)
        self._get_points = jax.jit(self._get_points, device=jax.devices(backend)[0])

    def __call__(
        self,
        latent_samples: jax.Array,
        labels: jax.Array,
        perplexity: float,
        iterations: int,
    ) -> PIL.Image.Image:
        points = self._get_points(
            latent_samples,
            perplexity=perplexity,
            iterations=iterations,
        )
        points = np.asarray(points)
        scatter = self._ax.scatter(
            points[:, 0],
            points[:, 1],
            c=self._color_map[labels],
            alpha=0.1,
        )
        try:
            image_bytes, size = self._canvas.print_to_buffer()
        finally:
            scatter.remove()
        image = PIL.Image.frombuffer("RGBA", size, image_bytes)
        return image

    def _get_points(
        self,
        latent_samples: jax.Array,
        perplexity: float,
        iterations: int,
    ) -> jax.Array:
        embeddings = algorithm.estimate_tsne(
            latent_samples,
            jax.random.PRNGKey(0),
            perplexity=perplexity,
            iterations=iterations,
            learning_rate=10.0,
            momentum=0.9,
        )
        return embeddings
