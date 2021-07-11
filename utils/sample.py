import numpy as np

def sample_mean(dist, win_size=5, rounds=1500):
    """Return the sampled mean distribution and mean value from the raw distribution

    Args:
        dist (ndarray): data distribution
        win_size (int): size of sampling window
        rounds (int): number of samplings
    """
    means = []
    for i in range(rounds):
        samples = np.random.choice(dist, size=win_size, replace=True)
        means.append(samples.mean())
    return means, np.array(means).mean()
