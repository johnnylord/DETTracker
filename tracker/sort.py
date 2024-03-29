import traceback
import numpy as np
from scipy.optimize import linear_sum_assignment
from track.base import TrackState
from track.motion.kalman2d import chi2inv95
from track.sort import SORTTrack
from track.utils.convert import tlwh_to_tlbr
from utils.preprocess import non_max_suppression


class SORT:

    def __init__(self,
                n_init=3,
                n_lost=3,
                n_dead=30,
                n_degree=4,
                iou_dist_threshold=0.3,
                nms_iou_threshold=1.0,
                **kwargs):
        # Track Management
        self.n_init = n_init
        self.n_lost = n_lost
        self.n_dead = n_dead
        # Tracker settings
        self.n_degree = n_degree
        self.iou_dist_threshold = iou_dist_threshold
        # Detection filtering
        self.nms_iou_threshold = nms_iou_threshold
        # Tracker state
        self.tracks = []

    def __call__(self, img, depthmap, flowmap, bboxes, masks):
        """Perform tracking on per frame basis

        Argument:
            img (tensor): tensor of shape (3, H, W)
            depthmap (tensor): tensor of shape (3, H, W)
            flowmap (tensor): tensor of shape (H, W, 2)
            bboxes (list): list of bounding boxes
            masks (list): list of object masks related to bboxes

        Format of depthmap:
            The value range of depth map is between 0 and 1, you can multiply
            a factor to get a relational estimated depth value.

        Format of flowmap:
            The value range of flow map is unbounded. In each pixel, there is
            a 2D xy pixel offset vector between consecutive frame.

        Format of bboxes:
            Each box in bboxes is represented as:
                (trackId, xmin, ymin, width, height, conf, 128 dim features...)
            (xmin, ymin , width, height) is in pixel coordinate

        Return:
            All objects being tracked represented as tracks. Each track has
            following information:
                1. Track ID
                2. bounding box
        """
        # Carry Track State from previous frame to current frame
        for track in self.tracks:
            track.predict()

        # Extract detected objects
        boxes = np.array([ tlwh_to_tlbr(box[1:1+4]) for box in bboxes ])# (N, 4)
        scores = np.array([ box[5] for box in bboxes ])                 # (N, 1)

        # Perform non-maximum suppression on detections
        picks = non_max_suppression(boxes, scores=scores, max_bbox_overlap=self.nms_iou_threshold)
        boxes = boxes[picks]

        if len(boxes) > 0:
            assert boxes.shape[-1] == 4
            observations = np.array(boxes)  # (N, 4)
        else:
            observations = np.array([])

        # No detected object in current frame
        if len(observations) == 0:
            for track in self.tracks:
                track.miss()
            self.tracks = [ track
                            for track in self.tracks
                            if track.state != TrackState.DEAD ]
            tracked = [ track.content
                        for track in self.tracks
                        if track.state == TrackState.TRACKED ]
            return tracked

        # Split track set by state
        conf_tracks = [ t for t in self.tracks if t.state != TrackState.TENTATIVE ]
        tent_tracks = [ t for t in self.tracks if t.state == TrackState.TENTATIVE ]

        # Associate detected objects with tracks
        match_pairs = []
        unmatch_tracks = []

        # Perform matching cascade on confirmed tracks
        pairs, tracks, observations = self._associate(
                                            conf_tracks,
                                            observations,
                                            threshold=self.iou_dist_threshold,
                                            mode='iou')
        match_pairs.extend(pairs)
        unmatch_tracks.extend(tracks)

        # Perform plain association with tentative tracks
        pairs, tracks, observations = self._associate(
                                            tent_tracks,
                                            observations,
                                            threshold=self.iou_dist_threshold,
                                            mode='iou')
        match_pairs.extend(pairs)
        unmatch_tracks.extend(tracks)

        # Update matched tracks and observations
        for track, observation in match_pairs:
            box = observation[:4]
            track.update(box)
            track.hit()

        # Update unmatching tracks
        for track in unmatch_tracks:
            track.miss()

        # Create new tracks
        for observation in observations:
            box = observation[:4]
            track = SORTTrack(box,
                            n_init=self.n_init,
                            n_lost=self.n_lost,
                            n_dead=self.n_dead)
            self.tracks.append(track)

        # Remove dead track
        self.tracks = [ track
                        for track in self.tracks
                        if track.state != TrackState.DEAD ]

        # Return tracked targets
        tracked = [ track.content
                    for track in self.tracks
                    if track.state == TrackState.TRACKED ]
        return tracked

    def _associate(self, tracks, observations, mode='iou', threshold=0.3):
        if len(tracks) == 0 and len(observations) != 0:
            return [], [], observations
        elif len(tracks) != 0 and len(observations) == 0:
            return [], tracks, np.array([])
        elif len(tracks) == 0 and len(observations) == 0:
            return [], [], np.array([])

        bboxes = observations[:, :4]

        # Concstruct cost matrix
        if mode == 'iou':
            cost_mat = np.array([ t.iou_dist(bboxes) for t in tracks ])
        else:
            raise ValueError("Unknown association mode")
        gate_mat = np.array([ t.square_maha_dist(bboxes, self.n_degree) for t in tracks ])
        cost_mat[gate_mat > chi2inv95[self.n_degree]] = 10000

        # Perform greedy matching algorithm
        tindices, oindices = linear_sum_assignment(cost_mat)
        match_pairs = [ pair
                        for pair in zip(tindices, oindices)
                        if cost_mat[pair[0], pair[1]] <= threshold ]

        # Prepare matching result
        pairs = [ (tracks[pair[0]], observations[pair[1]]) for pair in match_pairs ]

        unmatch_tindices = set(range(len(tracks))) - set([ pair[0] for pair in match_pairs ])
        unmatch_tindices = sorted(list(unmatch_tindices))
        unmatch_tracks = [ tracks[i] for i in unmatch_tindices ]

        unmatch_oindices = set(range(len(observations))) - set([ pair[1] for pair in match_pairs ])
        unmatch_oindices = sorted(list(unmatch_oindices))
        unmatch_observations = [ observations[i] for i in unmatch_oindices ]

        return pairs, unmatch_tracks, np.array(unmatch_observations)
