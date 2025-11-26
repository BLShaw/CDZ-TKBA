from collections import defaultdict
from src.config import Config

class ClusterCorrelation:
    def __init__(self, cluster, cdz):
        self.cdz = cdz
        self.cluster = cluster
        self.age = 1
        self.connections = defaultdict(int)
        self.cluster_objects = {}
        self.ref_clusters = []

    def update(self, q_packet, new_packet):
        assert q_packet.cluster == self.cluster
        assert q_packet.cortex != new_packet.cortex

        time_diff = (new_packet.time - q_packet.time)
        assert time_diff >= 0
        
        if time_diff < len(self.cdz.GAUSSIAN):
             temporal_weight = self.cdz.GAUSSIAN[time_diff]
        else:
             temporal_weight = 0

        correlation_update = Config.CE_LEARNING_RATE * temporal_weight * new_packet.strength * q_packet.strength

        self.connections[new_packet.cluster.name] += correlation_update
        self._normalize()
        self.age += 1

        self.cluster_objects[new_packet.cluster.name] = new_packet.cluster

    def _normalize(self, dict_to_normalize=None):
        if not dict_to_normalize:
            dict_to_normalize = self.connections

        val_sum = sum(dict_to_normalize.values())
        if val_sum > 0:
            for key, val in dict_to_normalize.items():
                dict_to_normalize[key] = val / val_sum

        return dict_to_normalize

    def remove_cluster(self, cluster):
        if cluster.name in self.connections:
            del self.connections[cluster.name]
        if cluster.name in self.cluster_objects:
            del self.cluster_objects[cluster.name]
        self._normalize()

    def add_ref(self, cluster):
        if cluster not in self.ref_clusters:
            self.ref_clusters.append(cluster)

    def get_strongest_correlation(self):
        if not self.connections:
            return None, 0
        cluster_name = max(self.connections, key=lambda conn_name: self.connections[conn_name])
        cluster = self.cluster_objects[cluster_name]
        strength = self.connections[cluster_name]
        return cluster, strength

    def uncertainty(self):
        cluster, strength = self.get_strongest_correlation()
        if not cluster:
            return 1.0
            
        feedback_scale = min(self.age / Config.CE_CERTAINTY_AGE_FACTOR, 1)
        certainty = (strength**2) * feedback_scale
        return 1 - certainty

    def certainty(self):
        return 1 - self.uncertainty()
