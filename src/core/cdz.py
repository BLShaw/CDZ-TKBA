from collections import deque
from collections import defaultdict
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
             from scipy.signal import windows
             gaussian = windows.gaussian(Config.CE_CORRELATION_WINDOW_MAX * 2, std=Config.CE_CORRELATION_WINDOW_STD, sym=True)

        self.GAUSSIAN = np.split(gaussian, 2)[0][::-1]

        if Config.CE_IGNORE_GAUSSIAN:
            self.GAUSSIAN *= 0
            self.GAUSSIAN[0] = 1

        self.PACKET_QUEUE_LENGTH = len(self.GAUSSIAN) + 1
        self.packet_queue = deque(maxlen=self.PACKET_QUEUE_LENGTH)
        self.correlations = {}
        
        # New: Track cluster firing frequency to penalize super-clusters
        self.cluster_frequencies = defaultdict(int)

    def update_competitive(self, v_cluster, a_cluster, penalty=1.0):
        """
        Explicit Competitive Hebbian Learning.
        Strengthens the V-A link.
        Actively WEAKENS the link between V and its OLD best A.
        Actively WEAKENS the link between A and its OLD best V.
        """
        # 1. Ensure Correlations exist
        if v_cluster.name not in self.correlations:
            self.correlations[v_cluster.name] = ClusterCorrelation(v_cluster, self)
        if a_cluster.name not in self.correlations:
            self.correlations[a_cluster.name] = ClusterCorrelation(a_cluster, self)
            
        # 2. Find existing strongest partners (The "Old Beliefs")
        v_conn = self.correlations[v_cluster.name]
        a_conn = self.correlations[a_cluster.name]
        
        old_a_best, _ = v_conn.get_strongest_correlation()
        old_v_best, _ = a_conn.get_strongest_correlation()
        
        # 3. Parameters
        lr = self.LEARNING_RATE * penalty
        
        # 4. Excite Current (Hebbian)
        v_conn.connections[a_cluster.name] += lr
        a_conn.connections[v_cluster.name] += lr
        
        # 5. Inhibit Old (Anti-Hebbian)
        # If V thought 'old_a_best' was the answer, but now sees 'a_cluster',
        # punish the old link to force a switch.
        if old_a_best and old_a_best.name != a_cluster.name:
            v_conn.connections[old_a_best.name] -= lr * 0.5 # Punish half as hard as we learn
            if v_conn.connections[old_a_best.name] < 0: v_conn.connections[old_a_best.name] = 0
            
        if old_v_best and old_v_best.name != v_cluster.name:
            a_conn.connections[old_v_best.name] -= lr * 0.5
            if a_conn.connections[old_v_best.name] < 0: a_conn.connections[old_v_best.name] = 0
            
        # 6. Normalize
        v_conn._normalize()
        a_conn._normalize()
        
        # 7. Update references
        v_conn.cluster_objects[a_cluster.name] = a_cluster
        a_conn.cluster_objects[v_cluster.name] = v_cluster

    def receive_packet(self, packet, learn=True):
        # Track frequency
        if learn:
            self.cluster_frequencies[packet.cluster.name] += 1
            
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

        # Calculate Frequency Penalty
        # Penalize the TARGET cluster if it fires too often.
        # This prevents everyone from connecting to the "Rich" cluster.
        target_freq = self.cluster_frequencies[new_packet.cluster.name]
        
        # Penalty factor: decay as 1 / sqrt(freq). Smoother than linear.
        penalty = 1.0 / max(1.0, np.sqrt(target_freq))
        
        self.correlations[old_packet.cluster.name].update(old_packet, new_packet, penalty=penalty)
        self.correlations[new_packet.cluster.name].add_ref(old_packet.cluster)

    def remove_cluster(self, cluster):
        if cluster.name in self.cluster_frequencies:
            del self.cluster_frequencies[cluster.name]
            
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