import numpy as np
from src.core.node import Node
from src.core.cluster import Cluster
from src.core.database import db
from src.config import Config
from src.core.core_utils import name_generator
from src.core.tkba_utils import cim

class NodeManager:
    def __init__(self, cortex, name=None):
        self.cortex = cortex
        self.name = name_generator(cortex, name)
        self.last_fired_node = None
        self.nn_index = None

    @property
    def nodes(self):
        return db.get_node_managers_nodes(self)

    def build_nrnd_index(self):
        pass

    def receive_encoding(self, encoding, learn=True):
        # Determine parameters based on cortex name
        if 'visual' in self.cortex.name.lower():
            sigma = Config.TKBA_VISUAL_SIGMA
            vigilance = Config.TKBA_VISUAL_VIGILANCE
        else:
            sigma = Config.TKBA_AUDIO_SIGMA
            vigilance = Config.TKBA_AUDIO_VIGILANCE

        # 1. Empty? Init
        if not self.nodes:
            self._add_node(encoding)
            new_node = self.nodes[-1]
            new_node.last_encoding = encoding
            cluster = new_node.get_strongest_cluster()
            cluster.excite_cdz(1.0, new_node, learn=learn)
            return cluster

        # 2. Calculate CIM (Distance)
        # Collect weights
        node_weights = np.array([n.position for n in self.nodes]) # (N, D)
        
        # Calculate distances
        dists = cim(encoding, node_weights, sigma) # (N,)
        
        # 3. Find Winner
        winner_idx = np.argmin(dists)
        min_dist = dists[winner_idx]
        winner_node = self.nodes[winner_idx]
        winner_node.last_encoding = encoding
        self.last_fired_node = winner_node
        
        # 4. Vigilance Test (TKBA Logic)
        if min_dist <= vigilance:
            # --- MATCH FOUND (Resonance) ---
            if learn:
                winner_node.learn(encoding)

            strongest_cluster = winner_node.get_strongest_cluster()
            strength = 1.0 
            strongest_cluster.excite_cdz(strength, winner_node, learn=learn)
            return strongest_cluster
            
        else:
            # --- MISMATCH (Novelty) ---
            # If max nodes not reached, create new
            if learn and len(self.nodes) < Config.MAX_NODES:
                # print(f"New Node created! Dist {min_dist:.4f} > {vigilance}")
                new_node, new_cluster = self._add_node(encoding)
                new_node.last_encoding = encoding
                self.last_fired_node = new_node
                
                new_cluster.excite_cdz(1.0, new_node, learn=learn)
                return new_cluster
            else:
                # Force fit if full (or not learning)
                winner_node.learn(encoding) if learn else None
                strongest_cluster = winner_node.get_strongest_cluster()
                strongest_cluster.excite_cdz(1.0, winner_node, learn=learn)
                return strongest_cluster

    def receive_feedback_packet(self, packet):
        if self.last_fired_node:
            self.last_fired_node.receive_feedback_packet(packet)

    def cleanup(self, delete_new_items=False):
        self._delete_underutilized_items()
        if delete_new_items:
            self._delete_new_items()

    def create_new_nodes(self):
        # TKBA creates nodes dynamically during 'receive_encoding' via Vigilance.
        # Keeping the old logic for splitting existing nodes if variance is high,
        # acting as a secondary growth mechanism.
        
        if len(self.nodes) >= Config.MAX_NODES:
            return

        num_nodes_added = 0
        sorted_nodes = sorted(self.nodes, key=lambda node: node.correlation_variance(), reverse=True)

        for node in sorted_nodes:
            if num_nodes_added >= Config.NODE_SPLIT_MAX_QTY: break
            if node.is_new() or node.is_underutilized(): continue
            
            # If correlation variance is high, splitting might help
            if node.correlation_variance() > Config.NODE_SPLIT_MAX_CORRELATION_VARIANCE:
                # Split
                clusters, strengths, positions, counts = db.get_nodes_clusters(node, include_all=True)
                
                # Create nodes for each sub-cluster
                created = False
                for idx, cluster in enumerate(clusters):
                    if counts[idx] > 3:
                        new_node, _ = self._add_node(positions[idx])
                        created = True
                        num_nodes_added += 1
                
                if created:
                    node.teardown()
                    num_nodes_added -= 1

    def _delete_underutilized_items(self):
        for node in self.nodes[:]:
            if node.is_underutilized():
                node.teardown()

    def _delete_new_items(self):
        for node in self.nodes[:]:
            if node.is_new():
                node.teardown()

    def _add_node(self, position):
        new_node = Node(self.cortex, position)
        new_cluster = Cluster(self.cortex, 'c_' + new_node.name)
        db.add_node(new_node, new_cluster)
        return new_node, new_cluster