
import numpy as np


def _get_knots_for_single_trajectory(
    t: np.ndarray, y: np.ndarray, n_internal_knots: int, s_guess: float | None = None
) -> tuple[np.ndarray, float]:
    from scipy.interpolate import UnivariateSpline

    tol_n_knots = 0
    tol_s = 1e-3
    s_lower_bound = 0.0
    s_upper_bound = None

    s = len(t) if s_guess is None else s_guess

    found_knots = UnivariateSpline(t, y, s=0).get_knots()
    if n_internal_knots >= len(found_knots):
        return found_knots, 0.0

    for _ in range(20):
        found_knots = UnivariateSpline(t, y, s=s).get_knots()
        if abs(len(found_knots) - n_internal_knots) <= tol_n_knots:
            return found_knots, s
        elif len(found_knots) < n_internal_knots:
            s_upper_bound = s
            s = (s + s_lower_bound) / 2
        elif len(found_knots) > n_internal_knots:
            s_lower_bound = s
            if s_upper_bound is None:
                s = 2 * s
            else:
                s = (s + s_upper_bound) / 2
        if s_upper_bound is not None:
            if s_upper_bound - s_lower_bound < tol_s:
                tol_n_knots += 1

    return found_knots, s


def calculate_knot_placement(
    ts: list[np.ndarray], ys: list[np.ndarray], n_internal_knots: int, T: float, seed: int = 0,
) -> np.ndarray:
    from sklearn.cluster import KMeans

    found_placements = []
    s_values = []
    for t_arr, y_arr in zip(ts, ys, strict=True):
        s_guess = float(np.mean(s_values)) if s_values else None
        knots, s = _get_knots_for_single_trajectory(t_arr, y_arr, n_internal_knots, s_guess)
        found_placements.append(knots)
        s_values.append(s)

    all_knots = np.concatenate(found_placements).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_internal_knots, random_state=seed).fit(all_knots)
    clusters = np.sort(kmeans.cluster_centers_.ravel())
    clusters[0] = 0.0
    clusters[-1] = T
    return clusters
