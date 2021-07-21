import scipy
import numpy as np

from .base import BaseTrack, TrackState
from .motion.kalman2d import KalmanFilter2D
from .utils.convert import tlbr_to_xyah, xyah_to_tlbr


class SORTTrack(BaseTrack):

    def __init__(self, box, **kwargs):
        super().__init__(**kwargs)
        # Motion Filter
        self.kf = KalmanFilter2D()
        # Initialize motion vector
        xyah = tlbr_to_xyah(box)
        self.mean, self.covar = self.kf.initiate(xyah)

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
        track = {
            'id': self.id,
            'state': state,
            'box': xyah_to_tlbr(self.mean[:4]),
            }
        return track

    def predict(self):
        """Carrry track state from t-1 to t timestamp"""
        mean, covar = self.kf.predict(self.mean, self.covar)
        self.mean = mean
        self.covar = covar
        return self.mean, self.covar

    def update(self, box):
        """Carrry use observation to calibrate track state at time t timestamp"""
        xyah = tlbr_to_xyah(box)
        mean, covar = self.kf.update(mean=self.mean,
                                    covariance=self.covar,
                                    observation=xyah)
        self.mean = mean
        self.covar = covar
        return self.mean, self.covar

    def iou_dist(self, bboxes):
        """Return iou distance vectors between track and bboxes

        Args:
            bboxes (np.ndarray): array of shape (N, 4)

        Return:
            A N dimensional iou distance vector

        Note:
            A bbox is (xmin, ymin, xmax, ymax)
        """
        bbox = np.array([xyah_to_tlbr(self.mean[:4])])
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

    def square_maha_dist(self, bboxes, n_degrees=4):
        """Return squared mahalanobis distance between track and bboxes

        Args:
            bboxes (np.ndarray): array of shape (N, 4)

        Return:
            A N dimensional distance vector

        Note:
            A bbox is (xmin, ymin, xmax, ymax)
        """
        xyahs = np.array([ tlbr_to_xyah(bbox) for bbox in bboxes ])
        mean, covar = self.kf.project(self.mean, self.covar)

        # Align number of dimensions
        mean, covar = mean[:n_degrees], covar[:n_degrees, :n_degrees]
        xyahs = xyahs[:, :n_degrees]

        # Apply mahalonobis distance formula
        cholesky_factor = np.linalg.cholesky(covar)
        d = xyahs - mean
        z = scipy.linalg.solve_triangular(cholesky_factor, d.T,
                                        check_finite=False,
                                        overwrite_b=True,
                                        lower=True)
        squared_maha = np.sum(z*z, axis=0)
        return squared_maha
