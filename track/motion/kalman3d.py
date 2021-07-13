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
        np.ndarray([
            x, y, z,            # 3D position
            vx, vy,             # xy optical flow pixel offset
            a, h,               # bounding box area
            vz, ax, ay, va, vh  # velocities
        ])

    Target explanation:
        x: bounding box center postion along x axis in image space
        y: bounding box center postion along y axis in image space
        z: bounding box center postion along z axis in pseudo depth space
        vx: object center pixel offset velocity along x axis in image space
        vy: object center pixel offset velocity along y axis in image space
        a: aspect ratio of bounding box width over bounding box height
        h: bounding box height
        [va]*: respective velocities of target states

    The motion model is a constant velocity model. The bounding box information
    (x, y, z, vx, vy, a, h) are taken as direct observation.

    Attributes:
        _motion_mat (ndarray): a (12, 12) matrix for predicting the next state
        _project_mat (ndarray): a (7, 12) matrix for projecting state vector from
            state space to observation space.
        _std_weight_pos (float): uncertainty of the (x, y, a, h)
        _std_weight_vel (float): uncertainty of the (vx, vy, va, vh)
        _std_weight_acc (float): uncertainty of the (ax, ay)
        _std_weight_dep (float): uncertainty of the (z, vz)
    """
    def __init__(self, max_depth=5, n_levels=30):
        self.max_depth = max_depth
        self.n_levels = n_levels

        # Dimension of prediction matrix: (12, 12)
        self._motion_mat = np.array([
            [ 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 ], # x
            [ 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 ], # y
            [ 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0 ], # z
            [ 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0 ], # vx
            [ 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0 ], # vy
            [ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 ], # a
            [ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 ], # h
            # ==========================================
            [ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 ], # vz
            [ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 ], # ax
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 ], # ay
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 ], # va
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 ], # vh
        ])

        # Dimension of projection matrix: (7, 12)
        self._project_mat = np.eye(7, 12)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimation. These weights control the amount of uncertainty in
        # the model. This is a bit hacky
        self._std_weight_pos = 1. / 20
        self._std_weight_vel = 1. / 160
        self._std_weight_acc = 1. / 640
        self._std_weight_dep = max_depth / n_levels / 100

    def initiate(self, observation):
        """Initialize state of kalman filter

        Args:
            observation (ndarray):
                Bounding box information (x, y, z, vx, vy, a, h) with
                center position (x, y, z),
                xy optical flow (vx, vy),
                aspect ratio (a),
                and height (h).

        Returns:
            mean (ndarray): intialized state vector of shape (12,)
            covariance (ndarray): initialized uncertainty matrix of shape (12x12)
        """
        # mean vector (state vector)
        mean_state = observation
        mean_veloc = np.array([0., 0., 0., 0., 0.])
        mean = np.concatenate([mean_state, mean_veloc])

        # covariance matrix (uncertainty matrix)
        std = [
            2 * self._std_weight_pos * observation[6],      # x ~= 50 pixel
            2 * self._std_weight_pos * observation[6],      # y ~= 50 pixel
            self.max_depth / self.n_levels,                 # z
            10 * self._std_weight_vel * observation[6],     # vx ~= 15 pixel
            10 * self._std_weight_vel * observation[6],     # vy ~= 15 pixel
            1e-2,                                           # a
            2 * self._std_weight_pos * observation[6],      # h ~= 50 pixel
            # =========================================================
            self._std_weight_dep,                           # vz
            10 * self._std_weight_acc * observation[6],     # ax ~= 2 pixel
            10 * self._std_weight_acc * observation[6],     # ay ~= 2 pixel
            1e-5,                                           # va
            10 * self._std_weight_vel * observation[6],     # vh ~= 15 pixel
            ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Predict the next state given the previous state (mean & covariance)

        Args:
            mean (ndarray): previous state vector of shape (12,)
            covariance (ndarray): previous uncertainty matrix of shape (12, 12)

        Returns:
            mean (ndarray): predicted state vector of shape (12,)
            covariance (ndarray): predicted uncertainty matrix of shape (12, 12)
        """
        # Noise for covariance matrix (noise from the "world", unknown factors)
        std = [
            self._std_weight_pos * mean[6],     # x ~= 50 pixel
            self._std_weight_pos * mean[6],     # y ~= 50 pixel
            self.max_depth / self.n_levels,     # z
            self._std_weight_vel * mean[6],     # vx ~= 15 pixel
            self._std_weight_vel * mean[6],     # vy ~= 15 pixel
            1e-2,                               # a
            self._std_weight_pos * mean[6],     # h ~= 50 pixel
            # =========================================================
            self._std_weight_dep,               # vz
            self._std_weight_acc * mean[6],     # ax ~= 2 pixel
            self._std_weight_acc * mean[6],     # ay ~= 2 pixel
            1e-5,                               # va
            self._std_weight_vel * mean[6],     # vh ~= 15 pixel
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
            mean (ndarray): predicted state vector of shape (12,)
            covariance (ndarray): predicted uncertainty matrix of shape (12, 12)

        Returns:
            mean (ndarray): projected state vector of shape (7,)
            covariance (ndarray): projected uncertainty matrix of shape (7, 7)
        """
        # Noise for projected covariance matrix
        std = [
            self._std_weight_pos * mean[6],     # x ~= 50 pixel
            self._std_weight_pos * mean[6],     # y ~= 50 pixel
            self.max_depth / self.n_levels,     # z
            self._std_weight_vel * mean[6],     # vx ~= 15 pixel
            self._std_weight_vel * mean[6],     # vy ~= 15 pixel
            1e-1,                               # a
            self._std_weight_pos * mean[6],     # h ~= 50 pixel
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
            mean (ndarray): predicted state vector of shape (12,)
            covariacne (ndarray): predicted uncertainty matrix of shape (12, 12)
            observation (ndarray): observed data of shape (7,)

        Returns:
            mean (ndarray): refined state vector of shape (12,)
            covariacne (ndarray): refined uncertainty matrix of shape (12, 12)
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
