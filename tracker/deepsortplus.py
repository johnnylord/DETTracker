import os
import os.path as osp
import traceback
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from scipy.optimize import linear_sum_assignment

from track.base import TrackState
from track.motion.kalman3d import chi2inv95
from track.deepsortplus import ContextTrack
from track.utils.convert import tlwh_to_tlbr
from utils.sample import sample_mean
from utils.function import softmax

from model.resnet import resnet50_reid
from data.dataset.motreid import MOTreID


class DeepSORTPlus:
    REID_CHECKPOINT = osp.join(osp.dirname(osp.dirname(__file__)), 'run/motreid/best.pth')

    def __init__(self,
                n_init=3,
                n_lost=3,
                n_dead=30,
                n_degree=3,
                n_levels=20,
                max_depth=5.,
                pool_size=100,
                iou_dist_threshold=0.3,
                cos_dist_threshold=0.3,
                maha_iou_dist_threshold=0.5,
                maha_cos_dist_threshold=0.5,
                device="cuda",
                guess_limit=1,
                indoor=False,
                **kwargs):
        self.indoor = indoor
        self.guess_limit = guess_limit
        # Track Management
        self.n_init = n_init
        self.n_lost = n_lost
        self.n_dead = n_dead
        # Tracker settings
        self.iou_dist_threshold = iou_dist_threshold
        self.cos_dist_threshold = cos_dist_threshold
        self.maha_iou_dist_threshold = maha_iou_dist_threshold
        self.maha_cos_dist_threshold = maha_cos_dist_threshold
        self.n_degree = n_degree
        self.n_levels = n_levels
        self.max_depth = max_depth
        self.pool_size = pool_size
        # Tracker state
        self.tracks = []
        # ReID Model
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = resnet50_reid(features=128, classes=1)
        checkpoint = torch.load(DeepSORTPlus.REID_CHECKPOINT, map_location={"cuda:1": "cuda:0"})
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        self.model.eval()
        # Transformation
        self.inverse = T.ToPILImage()
        self.transform = T.Compose([
                            T.Resize((MOTreID.IMG_HEIGHT, MOTreID.IMG_WIDTH)),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
                            ])

    def __call__(self, img, depthmap, flowmap, bboxes, masks):
        """Perform tracking on per frame basis

        Argument:
            img (tensor): tensor of shape (3, H, W)
            depthmap (tensor): tensor of shape (3, H, W)
            flowmap (tensor): tensor of shape (H, W, 2)
            bboxes (list): list of bounding boxes
            masks (list): list of object masks related bboxes

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
                3. motion vector
                4. depth
        """
        # Determine Track Occlusion status
        self._determine_track_occlusion()

        # ======================================================================
        # Extract detected objects
        oboxes, oconfs, omasks, ofeatures = [], [], [], []
        if len(masks) == 0:
            masks = [None]*len(bboxes)
        for box, mask in zip(bboxes, masks):
            if int(box[3])*int(box[4]) == 0:
                continue
            tlbr = tlwh_to_tlbr(box[1:1+4])
            conf = [box[5]]
            feature = box[-128:]
            assert len(tlbr) == 4 and len(conf) == 1 and len(feature) == 128
            oboxes.append(tlbr)
            oconfs.append(conf)
            omasks.append(mask)
            ofeatures.append(feature)

        # Convert to numpy foramt
        oboxes = np.array(oboxes)         # (N, 4)
        oconfs = np.array(oconfs)         # (N, 1)
        ofeatures = np.array(ofeatures)   # (N, 128)

        # Add depth information to boxes
        oboxes = self._add_depth(depthmap, oboxes, omasks) # (N, 5)

        # Aggregate observations
        if len(oboxes) > 0:
            observations = np.concatenate([ oboxes, oconfs, ofeatures ], axis=1)  # (N, 5+1+128)
        else:
            observations = np.array([])

        # ======================================================================
        # Carry Track State from previous frame to current frame
        for track in self.tracks:
            track.predict()

        # ======================================================================
        # No detected object in current frame
        if len(observations) == 0:
            for track in self.tracks:
                track.miss()
            self._filter_out_of_view(img)
            self.tracks = [ track
                            for track in self.tracks
                            if track.state != TrackState.DEAD ]
            tracked = [ track.content
                        for track in self.tracks
                        if track.state == TrackState.TRACKED ]
            return tracked

        # ======================================================================
        # Split track set by state
        live_tracks = [ t for t in self.tracks if t.state == TrackState.TRACKED ]
        lost_tracks = [ t for t in self.tracks if t.state == TrackState.LOST ]
        tent_tracks = [ t for t in self.tracks if t.state == TrackState.TENTATIVE ]

        # Associate detected objects with tracks
        match_pairs = []
        unmatch_tracks = []

        # Perform matching cascade with live tracks
        pairs, tracks, observations = self._matching_cascade(
                                            live_tracks,
                                            observations,
                                            threshold=self.maha_cos_dist_threshold,
                                            mode='maha_cos')
        match_pairs.extend(pairs)
        unmatch_tracks.extend([ track
                                for track in tracks
                                if not track.guessable ])
        # Try to compensate false positive detections
        guessable_tracks = [ track for track in tracks if track.guessable ]
        for track in guessable_tracks:
            boxes = np.array([ track.guess() ])
            _, features = self._extract_features(img, boxes[:, :4])
            if len(features) != 0:
                track.update(boxes[0])
                track.update_feature(features[0])

        # Perform matching cascade with lost tracks
        pairs, tracks, observations = self._matching_cascade(
                                            lost_tracks,
                                            observations,
                                            threshold=self.cos_dist_threshold,
                                            mode='cos')
        match_pairs.extend(pairs)
        unmatch_tracks.extend(tracks)

        # Perform matching cascade with tent tracks
        pairs, tracks, observations = self._associate(
                                            tent_tracks,
                                            observations,
                                            threshold=self.iou_dist_threshold,
                                            mode='iou')
        match_pairs.extend(pairs)
        unmatch_tracks.extend(tracks)

        # =====================================================================
        # Update matched tracks and observations
        for track, observation in match_pairs:
            # Unpack observation
            box = observation[:5]   # (xmin, ymin, xmax, ymax, depth)
            feature = observation[-128:]
            assert len(box) == 5 and len(feature) == 128
            track.update(box)
            track.update_feature(feature)
            track.hit()

        # Update unmatching tracks
        for track in unmatch_tracks:
            track.miss()

        # Create new tracks
        for observation in observations:
            # Unpack observation
            box = observation[:5]   # (xmin, ymin, xmax, ymax, depth)
            conf = observation[5]
            feature = observation[-128:]
            track = ContextTrack(box, feature,
                            n_levels=self.n_levels,
                            max_depth=self.max_depth,
                            pool_size=self.pool_size,
                            n_init=self.n_init,
                            n_lost=self.n_lost,
                            n_dead=self.n_dead,
                            indoor=self.indoor,
                            guess_limit=self.guess_limit)
            self.tracks.append(track)

        # Remove tracks that are out of view
        self._filter_out_of_view(img)

        # Remove dead track
        self.tracks = [ track
                        for track in self.tracks
                        if track.state != TrackState.DEAD ]

        # Return tracked targets
        tracked = [ track.content
                    for track in self.tracks
                    if track.state == TrackState.TRACKED ]

        return tracked

    def _matching_cascade(self, conf_tracks, observations, mode='cos', threshold=0.3):
        all_pairs = []
        all_tracks = []
        # Perform matching cascade association
        for priority in range(self.n_dead, 0, -1):
            ptracks = [ track
                        for track in conf_tracks
                        if track.priority == priority ]
            if len(ptracks) == 0:
                continue
            pairs, tracks, observations = self._associate(ptracks,
                                                    observations,
                                                    mode=mode,
                                                    threshold=threshold)
            all_pairs.extend(pairs)
            all_tracks.extend(tracks)

        return all_pairs, all_tracks, observations

    def _associate(self, tracks, observations, mode='cos', threshold=0.3):
        if len(tracks) == 0 and len(observations) != 0:
            return [], [], observations
        elif len(tracks) != 0 and len(observations) == 0:
            return [], tracks, np.array([])
        elif len(tracks) == 0 and len(observations) == 0:
            return [], [], np.array([])

        bboxes = observations[:, :5]        # (xmin, ymin, xmax, ymax, depth)
        features = observations[:, -128:]    # (features)
        assert bboxes.shape[1] == 5 and features.shape[1] == 128

        # Concstruct cost matrix
        if mode == 'cos':
            cost_mat = np.array([ t.cos_dist(features) for t in tracks ])
        elif mode == 'iou':
            cost_mat = np.array([ t.iou_dist(bboxes[:, :4]) for t in tracks ])
        elif mode == 'maha_cos':
            prob_cos = 1 - np.array([ t.cos_dist(features) for t in tracks ])
            # prob_dis = np.array([ -t.square_maha_dist(bboxes, n_degrees=self.n_degree) for t in tracks ])
            prob_dis = np.array([ -t.square_maha_dist(bboxes, n_degrees=2) for t in tracks ])
            prob_dis = np.array([ softmax(row) for row in prob_dis ])
            cost_mat = 1 - (prob_cos*prob_dis)
        elif mode == 'maha_iou':
            prob_iou = 1 - np.array([ t.iou_dist(bboxes[:, :4]) for t in tracks ])
            # prob_dis = np.array([ -t.square_maha_dist(bboxes, n_degrees=self.n_degree) for t in tracks ])
            prob_dis = np.array([ -t.square_maha_dist(bboxes, n_degrees=2) for t in tracks ])
            prob_dis = np.array([ softmax(row) for row in prob_dis ])
            cost_mat = 1 - (prob_iou*prob_dis)
        else:
            raise ValueError("Unknown mode '{mode}' for cost matrix")

        # Constrain cost matrix with gated matrix
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

    def _sample_mean(self, dist, win_size=5, rounds=100):
        """Return the sampled mean from the distribution

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

    def _add_depth(self, depthmap, boxes, masks):
        """Append depth (z) dimension to existing boxes

        Args:
            depthmap (tensor): tensor of shape (3, H, W)
            boxes (ndarray): bounding boxes of shape (N, 4)
            masks (list): list of object masks related to boxes

        Returns:
            a new set of boxes of shape (N, 5)

        NOTE:
            The box format of input is (xmin, ymin, xmax, ymax)
        """
        new_boxes = []
        for box, mask in zip(boxes, masks):
            if mask is not None:
                # Unpack bbox
                xmin = int(box[0])
                ymin = int(box[1])
                xmax = int(box[2])
                ymax = int(box[3])
                # Crop the pixels from depthmap
                pixels = depthmap[0, int(ymin):int(ymax), int(xmin):int(xmax)].numpy()
                if tuple(pixels.shape) != tuple(mask.shape):
                    mask = cv2.resize(mask, (pixels.shape[1], pixels.shape[0]))
                # Align pixels with mask
                pixels = pixels.reshape(-1)
                mask = mask.reshape(-1)
                mask = (mask > 200)
                # Filter out depth value
                depth = self.max_depth*(1-np.mean(pixels[mask]))
                new_boxes.append(box.tolist()+[depth])
            else:
                # Unpack bbox
                xmin = int(box[0]) if int(box[0]) >= 0 else 0
                ymin = int(box[1]) if int(box[1]) >= 0 else 0
                xmax = int(box[2]) if int(box[2]) <= depthmap.size(2) else depthmap.size(2)
                ymax = int(box[3]) if int(box[3]) <= depthmap.size(1) else depthmap.size(1)
                # Crop the pixels from depthmap
                dist = depthmap[0, ymin:ymax, xmin:xmax].numpy().reshape(-1)
                _, depth = self._sample_mean(dist)
                depth = self.max_depth*(1-depth)
                new_boxes.append(box.tolist()+[depth])

        return np.array(new_boxes)

    def _determine_track_occlusion(self):
        """Update confirmed tracks occlusion status"""
        tracked = [ track for
                    track in self.tracks
                    if track.state == TrackState.TRACKED ]
        confirm = [ track for
                    track in self.tracks
                    if track.state != TrackState.TENTATIVE ]
        for track in confirm:
            occluded = False
            for target in tracked:
                if target == track or target.depth >= track.depth:
                    continue
                xmin1, _, xmax1, _ = track.tlbr
                xmin2, _, xmax2, _ = target.tlbr
                xmin, xmax = max([xmin1, xmin2]), min([xmax1, xmax2])
                inter = max([xmax-xmin, 0])
                union = xmax1-xmin1
                if (inter/union) > 0.3:
                    occluded = True
                    break
            track.occluded = occluded

    def _extract_features(self, img, eboxes):
        """Extract 128-dimensional feature vectors for each eboxes

        Argument:
            img (tensor): torch tensor of shape (3, H, W)
            eboxes (ndarray): bounding box of shape (N, 4)

        Box format: (x1, y1, x2, y2)
        """
        boxes = []
        features = []
        for box in eboxes:
            xmin = int(box[0])
            ymin = int(box[1])
            xmax = int(box[2])
            ymax = int(box[3])
            try:
                crop = img[:, ymin:ymax, xmin:xmax]
                crop = self.inverse(crop)
                crop = self.transform(crop)
            except Exception as e:
                continue
            crop = crop.to(self.device)
            embed = self.model(crop.unsqueeze(0))[0]
            embed = embed.detach().cpu().numpy().tolist()
            # Save result
            features.append(embed)
            boxes.append(box.tolist())

        return np.array(boxes), np.array(features)

    def _filter_out_of_view(self, img):
        """Filter out tracks that are out of views"""
        tracks = []
        for track in self.tracks:
            box = track.tlbr
            box_width = box[2]-box[0]
            box_height = box[3]-box[1]
            # Check four cases
            if box[0] < 0: # (xmin)
                ratio = abs(box[0])/box_width
                if ratio > 0.3:
                    continue
            if box[1] < 0: # (ymin)
                ratio = abs(box[1])/box_height
                if ratio > 0.3:
                    continue
            if box[2] > img.shape[2]: # (xmax: img_width)
                ratio = abs(box[2]-img.shape[2])/box_width
                if ratio > 0.3:
                    continue
            if box[3] > img.shape[1]: # (ymax: img_height)
                ratio = abs(box[3]-img.shape[1])/box_height
                if ratio > 0.3:
                    continue
            tracks.append(track)
        self.tracks = tracks
