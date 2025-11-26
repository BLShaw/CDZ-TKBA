import numpy as np
from src.core.core_utils import name_generator
from src.core.database import db
from src.config import Config

class Node:
    def __init__(self, cortex, initial_position, name=None):
        self.name = name_generator(cortex, name)
        self.cortex = cortex
        self.created_at = self.cortex.timestep

        self.position = initial_position
        self.position_momentum = 0
        self.qty_feedback_packets = 0
        self.last_utilized = None
        self.last_encoding = None

    @property
    def age(self):
        return self.cortex.timestep - self.created_at

    def receive_feedback_packet(self, packet):
        amount = packet.strength * Config.NODE_TO_CLUSTER_LEARNING_RATE
        db.adjust_node_to_cluster_strength(self, packet.cluster, amount, self.last_encoding)
        self.qty_feedback_packets += 1

    def get_distance(self, position):
        distance_vector = self.position - position
        euclidean_distance = np.linalg.norm(distance_vector)
        return euclidean_distance, distance_vector

    def learn(self, position):
        error, distance_vector = self.get_distance(position)
        self._move_in_direction(-1 * distance_vector)
        self.last_utilized = self.cortex.timestep

    def _move_in_direction(self, direction):
        self.position += Config.NODE_POSITION_LEARNING_RATE * (direction + (Config.NODE_POSITION_MOMENTUM_ALPHA * self.position_momentum))
        self.position_momentum = (Config.NODE_POSITION_MOMENTUM_DECAY * self.position_momentum) + Config.NODE_POSITION_LEARNING_RATE * direction

    def is_underutilized(self):
        time_to_use = max(self.created_at, self.last_utilized) if self.last_utilized is not None else self.created_at
        return bool((self.cortex.timestep - time_to_use) >= Config.NODE_REQUIRED_UTILIZATION)

    def is_new(self):
        if self.last_utilized is None:
            return True
        if self.qty_feedback_packets <= Config.NODE_IS_NEW:
            return True
        return False

    def uncertainty(self):
        clusters, strengths = db.get_nodes_clusters(self, include_strengths=True)
        if not strengths:
            return 1.0
        
        feedback_scale = min(self.qty_feedback_packets / Config.NODE_CERTAINTY_AGE_FACTOR, 1)
        certainty = (max(strengths)**2) * feedback_scale
        return 1 - certainty

    def certainty(self):
        return 1 - self.uncertainty()

    def correlation_variance(self):
        clusters, strengths = db.get_nodes_clusters(self, include_strengths=True)
        if not strengths:
            return 1.0
        return 1 - max(strengths)

    def teardown(self):
        db.delete_node(self)

    def get_strongest_cluster(self):
        clusters, strengths = db.get_nodes_clusters(self, include_strengths=True)
        if not clusters:
            return None
        return clusters[np.argmax(strengths)]
