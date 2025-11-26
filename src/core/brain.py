from collections import deque
from src.config import Config
from src.core.cortex import Cortex
from src.core.cdz import CDZ
from src.core.database import db

class Brain:
    def __init__(self):
        self.timestep = 0
        self.cortices = {}
        self.output_stream = deque(maxlen=10)
        self.cdz = CDZ(self) # Initialized after assignment to self.cdz

    def add_cortex(self, cortex_name, autoencoder):
        if self.cortices.get(cortex_name):
            raise Exception('A cortex by this name is already present in this brain.')

        new_cortex = Cortex(self, cortex_name, autoencoder)
        self.cortices[cortex_name] = new_cortex
        return new_cortex

    def get_cortex(self, cortex_name):
        return self.cortices[cortex_name]

    def increment_timestep(self, amount=1):
        self.timestep += amount

    def receive_sensory_input(self, cortex, data, learn=True):
        return cortex.receive_sensory_input(data, learn=learn)

    def cleanup(self, force=False, delete_new_items=False):
        if force or delete_new_items or self.timestep % Config.BRN_CLEANUP_FREQUENCY == 0:
            print("====== Start cleanup =====")
            for cortex in self.cortices.values():
                cortex.cleanup(delete_new_items=delete_new_items)

            db.cleanup()
            self.build_nrnd_indexes(force=True)
            print("====== End cleanup ======")

    def create_new_nodes(self):
        if self.timestep % Config.BRN_NEURAL_GROWTH_FREQUENCY == 0:
            print("====== Start neural growth =====")
            for cortex in self.cortices.values():
                cortex.create_new_nodes()
            print("====== End neural growth =======")

    def build_nrnd_indexes(self, force=False):
        if force or self.timestep % Config.NRND_BUILD_FREQUENCY == 0:
            for cortex in self.cortices.values():
                cortex.node_manager.build_nrnd_index()
