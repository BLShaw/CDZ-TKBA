from collections import deque
import numpy as np
import scipy.signal as signal

from src.config import Config
from src.core.data_packet import DataPacket
from src.core.cluster_correlation import ClusterCorrelation

class CDZ:
    def __init__(self, brain):
        self.brain = brain
        self.LEARNING_RATE = Config.CE_LEARNING_RATE

        # Use signal.windows.gaussian if available, else fallback
        if hasattr(signal, 'windows') and hasattr(signal.windows, 'gaussian'):
             gaussian = signal.windows.gaussian(Config.CE_CORRELATION_WINDOW_MAX * 2, std=Config.CE_CORRELATION_WINDOW_STD, sym=True)
        elif hasattr(signal, 'gaussian'):
             gaussian = signal.gaussian(Config.CE_CORRELATION_WINDOW_MAX * 2, std=Config.CE_CORRELATION_WINDOW_STD, sym=True)
        else:
             # Fallback or raise
             # For many scipy versions it is in signal.windows
             from scipy.signal import windows
             gaussian = windows.gaussian(Config.CE_CORRELATION_WINDOW_MAX * 2, std=Config.CE_CORRELATION_WINDOW_STD, sym=True)

        self.GAUSSIAN = np.split(gaussian, 2)[0][::-1]

        if Config.CE_IGNORE_GAUSSIAN:
            self.GAUSSIAN *= 0
            self.GAUSSIAN[0] = 1

        self.PACKET_QUEUE_LENGTH = len(self.GAUSSIAN) + 1
        self.packet_queue = deque(maxlen=self.PACKET_QUEUE_LENGTH)
        self.correlations = {}

    def receive_packet(self, packet, learn=True):
        for q_packet in self.packet_queue:
            if (packet.time - q_packet.time) >= Config.CE_CORRELATION_WINDOW_MAX:
                break
            else:
                if packet.time == q_packet.time and learn:
                    self._update_connection(q_packet, packet)
                    self._update_connection(packet, q_packet)

                    self._send_feedback_packet(q_packet)
                    self._send_feedback_packet(packet)

        self._process_output(packet)
        self.packet_queue.appendleft(packet)

    def _process_output(self, packet):
        if self.correlations.get(packet.cluster.name):
            cluster, strength = self.correlations[packet.cluster.name].get_strongest_correlation()
            if cluster:
                node = cluster.get_strongest_node()
                self.brain.output_stream.appendleft(node)
            else:
                 self.brain.output_stream.appendleft(None)
        else:
            self.brain.output_stream.appendleft(None)

    def _send_feedback_packet(self, packet):
        if packet.cluster.name not in self.correlations:
            return

        cdz_connection = self.correlations[packet.cluster.name]
        cluster, cdz_strength = cdz_connection.get_strongest_correlation()
        
        if not cluster:
            return

        certainty_factor = packet.source_node.certainty() * cdz_connection.certainty()
        strength = (1 + certainty_factor)**2

        feedback_packet = DataPacket(cluster, strength, packet.time, packet.source_node)
        cluster.receive_feedback_packet(feedback_packet)

    def _update_connection(self, old_packet, new_packet):
        if new_packet.cortex == old_packet.cortex:
            return

        if not self.correlations.get(old_packet.cluster.name):
            self.correlations[old_packet.cluster.name] = ClusterCorrelation(old_packet.cluster, self)

        if not self.correlations.get(new_packet.cluster.name):
            self.correlations[new_packet.cluster.name] = ClusterCorrelation(new_packet.cluster, self)

        self.correlations[old_packet.cluster.name].update(old_packet, new_packet)
        self.correlations[new_packet.cluster.name].add_ref(old_packet.cluster)

    def remove_cluster(self, cluster):
        if self.correlations.get(cluster.name):
            excited_by = self.correlations[cluster.name].ref_clusters
            excites = self.correlations[cluster.name].cluster_objects.values()

            for e_cluster in excites:
                 if e_cluster.name in self.correlations:
                    if cluster in self.correlations[e_cluster.name].ref_clusters:
                         self.correlations[e_cluster.name].ref_clusters.remove(cluster)

            for excited_by_c in excited_by[:]:
                if excited_by_c.name in self.correlations:
                    self.correlations[excited_by_c.name].remove_cluster(cluster)

            del self.correlations[cluster.name]
