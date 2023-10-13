# Based on: https://nlml.github.io/in-raw-numpy/in-raw-numpy-t-sne/
import jax
import jax.numpy as jnp


def neg_squared_euc_dists(x: jax.Array):
    sum_x = jnp.sum(jnp.square(x), 1)
    distances = jnp.add(jnp.add(-2 * jnp.dot(x, x.T), sum_x).T, sum_x)
    return -distances


def calc_prob_matrix(distances: jax.Array, sigmas: jax.Array) -> jax.Array:
    two_sig_sq = 2.0 * jnp.square(sigmas.reshape((-1, 1)))
    return jax.nn.softmax(distances / two_sig_sq)


def p_conditional_to_joint(p: jax.Array) -> jax.Array:
    return (p + p.T) / (2.0 * p.shape[0])


def p_joint(x: jax.Array, target_perplexity: jax.Array) -> jax.Array:
    distances = neg_squared_euc_dists(x)
    sigmas = find_optimal_sigmas(distances, target_perplexity)
    p_conditional = calc_prob_matrix(distances, sigmas)
    p = p_conditional_to_joint(p_conditional)
    return p


def binary_search(
    func,
    index: int,
    target_perplexity: float,
    tolerance: float = 1.0e-3,
    max_iterations: int = 10000,
    low=1e-20,
    high=1000.0,
) -> jax.Array:
    def cond(
        state: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]
    ) -> bool:
        iteration, low, high, midpoint, value = state
        return jnp.logical_and(
            jnp.abs(value - target_perplexity) > tolerance, (iteration < max_iterations)
        )

    def body(
        state: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        iteration, low, high, midpoint, value = state
        midpoint = 0.5 * (low + high)
        value = func(index, midpoint)
        update_upper = value - target_perplexity > 0
        low = jnp.where(update_upper, low, midpoint)
        high = jnp.where(update_upper, midpoint, high)
        return (iteration + 1, low, high, midpoint, value)

    intial_state = (0, low, high, jnp.inf, jnp.inf)

    iteration, low, high, midpoint, value = jax.lax.while_loop(cond, body, intial_state)
    return 0.5 * (low + high)


def calculate_perplexity(prob_matrix: jax.Array) -> jax.Array:
    entropy = -jnp.sum(prob_matrix * jnp.log2(prob_matrix), 1)
    perplexity = 2**entropy
    return perplexity


def perplexity(distances: jax.Array, sigmas: jax.Array) -> jax.Array:
    return calculate_perplexity(calc_prob_matrix(distances, sigmas))


def find_optimal_sigma(
    distances: jax.Array, target_perplexity: float, index: int
) -> jax.Array:
    eval_fn = lambda index, sigma: perplexity(
        distances[[index], :],
        sigma,
    ).squeeze()
    return binary_search(eval_fn, index, target_perplexity)


def find_optimal_sigmas(
    distances: jax.Array, target_perplexity: jax.Array
) -> jax.Array:
    sigmas = jax.vmap(find_optimal_sigma, in_axes=(None, None, 0))(
        distances, target_perplexity, jnp.arange(distances.shape[0])
    )
    return sigmas


def q_tsne(y: jax.Array) -> jax.Array:
    distances = neg_squared_euc_dists(y)
    inv_distances = jnp.power(1.0 - distances, -1)
    diag_elements = jnp.diag_indices_from(inv_distances)
    inv_distances = inv_distances.at[diag_elements].set(0.0)
    return inv_distances / jnp.sum(inv_distances), inv_distances


def tsne_grad(
    p: jax.Array, q: jax.Array, y: jax.Array, inv_distances: jax.Array
) -> jax.Array:
    pq_diff = p - q
    pq_expanded = jnp.expand_dims(pq_diff, 2)
    y_diffs = jnp.expand_dims(y, 1) - jnp.expand_dims(y, 0)

    distances_expanded = jnp.expand_dims(inv_distances, 2)
    y_diffs_wt = y_diffs * distances_expanded

    grad = 4.0 * (pq_expanded * y_diffs_wt).sum(1)
    return grad


def estimate_tsne(
    x: jax.Array,
    key: jax.random.KeyArray,
    perplexity: float,
    iterations: int,
    learning_rate: float,
    momentum: float,
) -> jax.Array:
    p = p_joint(x, perplexity)

    y = 0.0 + 0.0001 * jax.random.normal(key, shape=(x.shape[0], 2))

    y_m2 = y
    y_m1 = y

    def body(
        i: int, val: tuple[jax.Array, jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        del i
        y, y_m1, y_m2 = val
        q, distances = q_tsne(y)
        grads = tsne_grad(p, q, y, distances)
        y = y - learning_rate * grads
        y += momentum * (y_m1 - y_m2)
        y_m2 = y_m1
        y_m1 = y
        return (y, y_m1, y_m2)

    init_val = (y, y_m1, y_m2)

    y, y_m1, y_m2 = jax.lax.fori_loop(0, iterations, body, init_val)

    return y
