import numpy as np
import torch
from src.core.node_manager import NodeManager
from src.core.database import db

class Cortex:
    def __init__(self, brain, name, autoencoder):
        self.name = name
        self.autoencoder = autoencoder
        self.brain = brain

        self.node_manager = NodeManager(self)
        db.node_manager_to_nodes.add(self.node_manager, [], [])

    @property
    def cdz(self):
        return self.brain.cdz

    @property
    def timestep(self):
        return self.brain.timestep

    def receive_sensory_input(self, data, learn=True):
        """
        Accepts data. 
        If data is a torch Tensor, it passes it through the autoencoder to get the encoding.
        If data is a numpy array, it assumes it is already an encoding.
        """
        encoding = None
        
        if isinstance(data, torch.Tensor):
            # Ensure it's on the right device for the model
            if hasattr(self.autoencoder, 'get_encoding'):
                with torch.no_grad():
                    # Ensure correct device
                    device = next(self.autoencoder.parameters()).device
                    if data.device != device:
                        data = data.to(device)
                        
                    encoded_tensor = self.autoencoder.get_encoding(data)
                    encoding = encoded_tensor.cpu().numpy().flatten()
            else:
                # No encoder, assume raw tensor is the encoding
                encoding = data.cpu().numpy().flatten()
                
        elif isinstance(data, np.ndarray):
            encoding = data.flatten()
            
        else:
            raise ValueError(f"Unsupported data type for cortex input: {type(data)}")

        strongest_cluster = self.node_manager.receive_encoding(encoding, learn=learn)
        return strongest_cluster

    def cleanup(self, delete_new_items=False):
        self.node_manager.cleanup(delete_new_items=delete_new_items)

    def create_new_nodes(self):
        self.node_manager.create_new_nodes()