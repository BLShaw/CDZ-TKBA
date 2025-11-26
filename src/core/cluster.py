import numpy as np
from src.core.core_utils import name_generator
from src.core.database import db
from src.core.data_packet import DataPacket
from src.config import Config

class Cluster:
    def __init__(self, cortex, name, required_utilization=Config.CLUSTER_REQUIRED_UTILIZATION):
        self.name = name_generator(cortex, name)
        self.cortex = cortex
        self.created_at = self.cortex.timestep
        self.last_fired = self.created_at
        self.last_feedback_packet = self.created_at
        self.REQUIRED_UTILIZATION = required_utilization

    @property
    def age(self):
        return self.timestep - self.created_at

    @property
    def nodes(self):
        return db.get_clusters_nodes(self)

    @property
    def cdz(self):
        return self.cortex.brain.cdz

    @property
    def node_manager(self):
        return self.cortex.node_manager

    @property
    def timestep(self):
        return self.cortex.brain.timestep

    def excite_cdz(self, strength, source_node, learn=True):
        packet = DataPacket(self, strength, self.timestep, source_node)
        self.last_fired = self.cortex.timestep

        if learn:
            amount = Config.CLUSTER_NODE_LEARNING_RATE
            db.adjust_cluster_to_node_strength(self, source_node, amount)

        self.cdz.receive_packet(packet, learn=learn)

    def is_underutilized(self):
        time_to_use = max(self.created_at, self.last_fired, self.last_feedback_packet)
        return bool((self.cortex.timestep - time_to_use) >= self.REQUIRED_UTILIZATION)

    def receive_feedback_packet(self, feedback_packet):
        self.last_feedback_packet = self.cortex.timestep
        self.node_manager.receive_feedback_packet(feedback_packet)

    def get_strongest_node(self):
        nodes, strengths = db.get_clusters_nodes(self, include_strengths=True)
        if not nodes:
            return None
        return nodes[np.argmax(strengths)]
