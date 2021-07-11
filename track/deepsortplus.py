import scipy
import numpy as np

from .base import BaseTrack, TrackState
from .motion.kalman3d import KalmanFilter3D
from .utils.convert import tlbr_to_xyah, xyah_to_tlbr


class ContextTrack(BaseTrack):
    """Track object in DeepSORTPlus

    Arguments:
        box (ndarray): motion information (xmin, ymin, xmax, ymax, z, vx, vy)
        feature (ndarray): ReID feature vector of 128 dimension
        pool_size (ndarray): size of feature pool
        max_depth (float): maximum depth of virtual space
        n_levels (int): number of depth levels in virtual space
    """
    def __init__(self, box, feature, pool_size, max_depth, n_levels, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size
        # Motion Filter
        self.kf = KalmanFilter3D(max_depth=max_depth, n_levels=n_levels)
        # Initialize motion vector
        x, y, a, h = tlbr_to_xyah(box[:4])
        z, vx, vy = box[4:].tolist()
        motion = np.array([ x, y, z, vx, vy, a, h ])
        self.mean, self.covar = self.kf.initiate(motion)
        # initilaize reid feature sets
        self.feature_pool = [ feature ]

    @property
    def content(self):
        if self.state == TrackState.TRACKED:
            state = "tracked"
        elif self.state == TrackState.LOST:
            state = "lost"
        elif self.state == TrackState.TENTATIVE:
            state = "tentative"
        else:
            state = "dead"
        # Content of track
        mean, covar = self.kf.project(self.mean, self.covar)
        box = xyah_to_tlbr([ mean[0], mean[1], mean[5], mean[6] ])
        xyz = mean[:3]
        var = covar[:3, :3]
        motion = mean[3:5]
        track = {
            'id': self.id,
            'state': state,
            'box': box,
            'xyz': xyz,
            'var': var,
            'motion': motion,
            }
        return track

    def predict(self):
        """Carrry track state from t-1 to t timestamp"""
        mean, covar = self.kf.predict(self.mean, self.covar)
        self.mean = mean
        self.covar = covar
        return self.mean, self.covar

    def update(self, box):
        """Use observation to calibrate track state at time t timestamp"""
        # Convert observation format
        x, y, a, h = tlbr_to_xyah(box[:4])
        z, vx, vy = box[4:].tolist()
        observation = np.array([ x, y, z, vx, vy, a, h ])
        # Update state with observatio
        mean, covar = self.kf.update(mean=self.mean,
                                    covariance=self.covar,
                                    observation=observation)
        self.mean = mean
        self.covar = covar
        return self.mean, self.covar

    def update_feature(self, feature):
        """Update ReID feature pool"""
        self.feature_pool.append(feature)
        if len(self.feature_pool) >= self.pool_size:
            self.feature_pool = self.feature_pool[1:]

    def iou_dist(self, bboxes):
        """Return iou distance vectors between track and bboxes

        Args:
            bboxes (np.ndarray): array of shape (N, 4)

        Return:
            A N dimensional iou distance vector

        Note:
            A bbox is (xmin, ymin, xmax, ymax, z, vx, vy)
        """
        xyah = [ self.mean[0], self.mean[1], self.mean[5], self.mean[6] ]
        bbox = np.array([xyah_to_tlbr(xyah)])

        x11, y11, x12, y12 = np.split(bbox, 4, axis=1)
        x21, y21, x22, y22 = np.split(bboxes, 4, axis=1)

        xA = np.maximum(x11, np.transpose(x21))
        yA = np.maximum(y11, np.transpose(y21))
        xB = np.minimum(x12, np.transpose(x22))
        yB = np.minimum(y12, np.transpose(y22))

        interArea = np.maximum((xB-xA+1), 0)*np.maximum((yB-yA+1), 0)
        bbox1Area = (x12-x11+1)*(y12-y11+1)
        bbox2Area = (x22-x21+1)*(y22-y21+1)

        iou = interArea / (bbox1Area+np.transpose(bbox2Area)-interArea)
        return (1 - iou).reshape(-1)

    def cos_dist(self, features):
        """Return cosine distance between features and feature pool of track

        Args:
            features (np.ndarray): array of shape (N, feature_dim)

        Return:
            A N dimensional cosine distance vecotr

        Note:
            Each feature vector is a unit vector
        """
        feature_pool = np.array(self.feature_pool)
        cosines = np.dot(feature_pool, features.T)
        return (1 - cosines).min(axis=0)

    def square_maha_dist(self, bboxes):
        """Return squared mahalanobis distance between track and bboxes

        Args:
            bboxes (np.ndarray): array of shape (N, 7)

        Return:
            A N dimensional distance vector

        Note:
            A bbox is (xmin, ymin, xmax, ymax, z, vx, vy)
        """
        # Convert bboxes format to state vector
        observations = []
        for bbox in bboxes:
            x, y, a, h = tlbr_to_xyah(bbox[:4])
            z, vx, vy = bbox[4:].tolist()
            motion = np.array([ x, y, z, vx, vy, a, h ])
            observations.append(motion)
        observations = np.array(observations)

        # Align dimension of state vector to observations
        mean, covar = self.kf.project(self.mean, self.covar)

        # Apply mahalonobis distance formula
        cholesky_factor = np.linalg.cholesky(covar)
        d = observations - mean
        z = scipy.linalg.solve_triangular(cholesky_factor, d.T,
                                        check_finite=False,
                                        overwrite_b=True,
                                        lower=True)
        squared_maha = np.sum(z*z, axis=0)
        return squared_maha
