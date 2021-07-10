from abc import ABC, abstractmethod


__all__ = [ "TrackState", "TrackAction", "BaseTrack" ]


class TrackState:
    """Helper class containing all track state"""
    TENTATIVE = 0
    TRACKED = 1
    LOST = 2
    DEAD = 3


class TrackAction:
    """Helper class containing actions can performed on tracks"""
    HIT = 0
    MISS = 1


class BaseTrack(ABC):
    """Base class for all kind of derived track class

    State transition machine:
                              [hit]
                            +------+
                            v      |
    3 continous [hit]     +--------+-+    5 continuous [miss]
            +------------>| Tracked  +-------------+
            |             +----------+             v   +----+
       +----+-----+             ^             +--------++   |
    -->| Tentative|             +-------------+ Lost    |  [miss]
       +----+-----+                  1 [hit]  +----+----+   |
            |                                      |   ^    |
            |             +----------+             |   +----+
            +------------>+  Dead    |<------------+
            1 [miss]      +----------+      exceed 30 [miss]
    Attributes:
        id (int): id number of the track
        state (int): current state of the track
    """
    TRACK_COUNTER = 0

    def __init__(self, n_init=3, n_lost=5, n_dead=30):
        # State transition threshold
        self.n_init = n_init
        self.n_lost = n_lost
        self.n_dead = n_dead
        # Public members
        self.id = self.TRACK_COUNTER
        self.state = TrackState.TENTATIVE
        self.priority = self.n_dead
        # Private members
        self._hit_count = 1
        self._miss_count = 0
        self._recent_actions = []
        # Update static variable
        self.TRACK_COUNTER += 1

    def hit(self):
        # Renew priortiy level
        self.priority = self.n_dead
        # Update Private members
        self._hit_count += 1
        self._recent_actions.append(TrackAction.HIT)
        if len(self._recent_actions) > self.n_dead:
            self._recent_actions = self._recent_actions[1:]
        # Update Track State
        if (
            self.state == TrackState.TENTATIVE
            and self._hit_count >= self.n_init
        ):
            self.state = TrackState.TRACKED
        elif (
            self.state == TrackState.TENTATIVE
            and self._hit_count < self.n_init
        ):
            self.state = TrackState.TENTATIVE
        else:
            self.state = TrackState.TRACKED

    def miss(self):
        # Update priority level
        self.priority -= 1
        # Update Private members
        self._miss_count += 1
        self._recent_actions.append(TrackState.MISS)
        if len(self._recent_actions) > self.n_dead:
            self._recent_actions = self._recent_actions[1:]
        # Update Track State
        if self.state == TrackState.TENTATIVE:
            self.state = TrackState.DEAD
        elif (
            self.state == TrackState.TRACKED
            and len([
                action
                for action in self._recent_actions[-self.n_lost:]
                if action == TrackAction.MISS ]) >= self.n_lost
        ):
            self.state = TrackState.LOST
        elif (
            self.state == TrackState.LOST
            and self.priority <= 0
        ):
            self.state = TrackState.DEAD

    @abstractmethod
    def predict(self, *args, **kwargs):
        raise RuntimeError("You should not call predict in abstract classs")

    @abstractmethod
    def update(self, *args, **kwargs):
        raise RuntimeError("You should not call update in abstract classs")
