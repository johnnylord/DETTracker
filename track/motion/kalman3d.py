import numpy as np
import scipy.linalg


__all__ = [ "KalmanFilter3D", "chi2inv95" ]

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter3D:
    """Kalman filter for modeling spatial state of track in image space

    Here are the states kalman filter trying to maintain

        np.ndarray([ x, y, z, a, h, vx, vy, vz, va, vh ])

    Target explanation:
        x: bounding box center postion along x axis in image space
        y: bounding box center postion along y axis in image space
        z: bounding box center postion along z axis in pseudo depth space
        a: aspect ratio of bounding box width over bounding box height
        h: bounding box height
        [va]*: respective velocities of target states

    The motion model is a constant velocity model. The bounding box information
    (x, y, z, a, h) are taken as direct observation.

    Attributes:
        _motion_mat (ndarray): a (10, 10) matrix for predicting the next state
        _project_mat (ndarray): a (5, 10) matrix for projecting state vector from
            state space to observation space.
        _std_weight_pos (float): uncertainty of the (x, y, a, h)
        _std_weight_vel (float): uncertainty of the (vx, vy, va, vh)
        _std_weight_dep (float): uncertainty of the (z)
        _std_weight_dev (float): uncertainty of the (vz)
    """
    def __init__(self, max_depth=5, n_levels=20):
        self.max_depth = max_depth
        self.n_levels = n_levels

        # Dimension of prediction matrix: (12, 12)
        self._motion_mat = np.array([
            [ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0 ], # x
            [ 0, 1, 0, 0, 0, 0, 1, 0, 0, 0 ], # y
            [ 0, 0, 1, 0, 0, 0, 0, 1, 0, 0 ], # z
            [ 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 ], # a
            [ 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 ], # h
            [ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 ], # vx
            [ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 ], # vy
            [ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 ], # vz
            [ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 ], # va
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 ], # vh
        ])

        # Dimension of projection matrix: (5, 10)
        self._project_mat = np.eye(5, 10)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimation. These weights control the amount of uncertainty in
        # the model. This is a bit hacky
        self._std_weight_pos = 1. / 20
        self._std_weight_vel = 1. / 160
        self._std_weight_dep = max_depth / n_levels
        self._std_weight_dev = max_depth / n_levels / 1000

    def initiate(self, observation):
        """Initialize state of kalman filter

        Args:
            observation (ndarray):
                Bounding box information (x, y, z, a, h) with
                center position (x, y, z),
                aspect ratio (a), and height (h).

        Returns:
            mean (ndarray): intialized state vector of shape (10,)
            covariance (ndarray): initialized uncertainty matrix of shape (10x10)
        """
        # mean vector (state vector)
        mean_state = observation
        mean_veloc = np.array([0., 0., 0., 0., 0.])
        mean = np.concatenate([mean_state, mean_veloc])

        # covariance matrix (uncertainty matrix)
        std = [
            2 * self._std_weight_pos * observation[4],
            2 * self._std_weight_pos * observation[4],
            2 * self._std_weight_dep,
            1e-2,
            2 * self._std_weight_pos * observation[4],
            # =========================================
            10 * self._std_weight_vel * observation[4],
            10 * self._std_weight_vel * observation[4],
            10 * self._std_weight_dev,
            1e-5,
            10 * self._std_weight_vel * observation[4],
            ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Predict the next state given the previous state (mean & covariance)

        Args:
            mean (ndarray): previous state vector of shape (10,)
            covariance (ndarray): previous uncertainty matrix of shape (10, 10)

        Returns:
            mean (ndarray): predicted state vector of shape (10,)
            covariance (ndarray): predicted uncertainty matrix of shape (10, 10)
        """
        # Noise for covariance matrix (noise from the "world", unknown factors)
        std = [
            self._std_weight_pos * mean[4],
            self._std_weight_pos * mean[4],
            self._std_weight_dep,
            1e-2,
            self._std_weight_pos * mean[4],
            # =============================
            self._std_weight_vel * mean[4],
            self._std_weight_vel * mean[4],
            self._std_weight_dev,
            1e-5,
            self._std_weight_vel * mean[4],
            ]
        motion_cov = np.diag(np.square(np.array(std)))

        # Update mean vector and covariance matrix
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """Project state vector and uncertainty matrix to the observation space

        Args:
            mean (ndarray): predicted state vector of shape (10,)
            covariance (ndarray): predicted uncertainty matrix of shape (10, 10)

        Returns:
            mean (ndarray): projected state vector of shape (5,)
            covariance (ndarray): projected uncertainty matrix of shape (5, 5)
        """
        # Noise for projected covariance matrix
        std = [
            self._std_weight_pos * mean[4],
            self._std_weight_pos * mean[4],
            self._std_weight_dep,
            1e-1,
            self._std_weight_pos * mean[4],
            ]
        project_cov = np.diag(np.square(std))

        # Projected mean and covariance matrix
        mean = np.dot(self._project_mat, mean)
        covariance = np.linalg.multi_dot((
            self._project_mat, covariance, self._project_mat.T)) + project_cov

        return mean, covariance

    def update(self, mean, covariance, observation):
        """Refine the predicted state with the observed data

        Args:
            mean (ndarray): predicted state vector of shape (10,)
            covariacne (ndarray): predicted uncertainty matrix of shape (10, 10)
            observation (ndarray): observed data of shape (5,)

        Returns:
            mean (ndarray): refined state vector of shape (10,)
            covariacne (ndarray): refined uncertainty matrix of shape (10, 10)
        """
        # Project mean & covariance so that they are in the same space as observation
        project_mean, project_covariance = self.project(mean, covariance)

        # Calculate kalman gain
        chol_factor, lower = scipy.linalg.cho_factor(
                                    project_covariance,
                                    lower=True,
                                    check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
                                    (chol_factor, lower),
                                    np.dot(covariance, self._project_mat.T).T,
                                    check_finite=False).T

        # Update mean and covariance with observation
        mean = mean + np.dot(observation-project_mean, kalman_gain.T)
        covariance = covariance - np.linalg.multi_dot((
            kalman_gain, project_covariance, kalman_gain.T))

        return mean, covariance
