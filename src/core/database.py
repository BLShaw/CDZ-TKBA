import numpy as np

class BasicTable:
    def __init__(self, name):
        self.name = name
        self.data = {}

    def add(self, item):
        self.data[item.name] = item

    def remove(self, item):
        del self.data[item.name]

    def get(self, item_name):
        return self.data[item_name]
    
    def verify_data_integrity(self):
        pass

class OneToManyTable:
    def __init__(self, name):
        self.name = name
        self.data = {}

    def add(self, item, related_items, strengths, position=None):
        if item.name not in self.data:
            if not isinstance(related_items, list) or not isinstance(strengths, list):
                raise ValueError("related_items and strengths must be lists")
            
            self.data[item.name] = {
                'obj': item,
                'list': related_items,
                'strengths': strengths,
                'position': [position],
                'count': [1]
            }
        else:
            raise Exception('Item already in table')

    def get(self, item):
        return self.data[item.name]

    def remove(self, item):
        if item.name in self.data:
            del self.data[item.name]

    def add_related_item(self, item, related_item, strength=1.0, position=None):
        if item.name not in self.data:
            self.add(item, [related_item], [1.0], position=position)
        else:
            entry = self.data[item.name]
            if related_item not in entry['list']:
                entry['list'].append(related_item)
                entry['strengths'].append(strength)
                entry['position'].append(position)
                entry['count'].append(1)
            else:
                raise Exception('The item already is related.')
        
        self._normalize_item(item)

    def remove_related_item(self, item, related_item):
        entry = self.data[item.name]
        if related_item in entry['list']:
            idx = entry['list'].index(related_item)
            entry['list'].remove(related_item)
            entry['strengths'].pop(idx)
            entry['position'].pop(idx)
            entry['count'].pop(idx)

            if len(entry['list']) > 0:
                self._normalize_item(item)

    def is_related(self, item, related_item):
        if item.name not in self.data:
            return False
        return related_item in self.data[item.name]['list']

    def increase_relationship_strength(self, item, related_item, amount, position=None):
        if not self.is_related(item, related_item):
            raise Exception("Items are not related")

        entry = self.data[item.name]
        idx = entry['list'].index(related_item)
        entry['strengths'][idx] += amount
        entry['count'][idx] += 1

        if entry['position'][idx] is not None and position is not None:
            old_pos = entry['position'][idx]
            # Simple moving average approximation
            new_pos = old_pos + (position - old_pos) / entry['count'][idx]
            entry['position'][idx] = new_pos
        elif entry['position'][idx] is None:
            entry['position'][idx] = position

        self._normalize_item(item)

    def _normalize_item(self, item):
        entry = self.data[item.name]
        total = sum(entry['strengths'])
        if total <= 0.0:
            # Avoid division by zero
            return
        entry['strengths'] = [s / total for s in entry['strengths']]

    def verify_data_integrity(self):
        for val in self.data.values():
            assert len(val['list']) == len(val['strengths'])

class Database:
    def __init__(self):
        self.nodes = BasicTable('nodes')
        self.clusters = BasicTable('clusters')
        self.nodes_to_clusters = OneToManyTable('nodes_to_clusters')
        self.clusters_to_nodes = OneToManyTable('clusters_to_nodes')
        self.node_manager_to_nodes = OneToManyTable('node_manager_to_nodes')

    def add_node(self, node, cluster, initial=False):
        # print(f">> adding node: {node.name} {'(initial)' if initial else ''}")
        self.nodes.add(node)
        self.clusters.add(cluster)
        
        self.nodes_to_clusters.add(node, [cluster], [1.0])
        self.clusters_to_nodes.add(cluster, [node], [1.0])
        self.node_manager_to_nodes.add_related_item(node.cortex.node_manager, node)

    def delete_node(self, node):
        # print(f">> removing node: {node.name}")
        self.nodes.remove(node)
        
        # Remove relationships
        clusters = self.get_nodes_clusters(node)
        for cluster in clusters:
            self.clusters_to_nodes.remove_related_item(cluster, node)
            
        self.nodes_to_clusters.remove(node)
        self.node_manager_to_nodes.remove_related_item(node.cortex.node_manager, node)

    def _delete_cluster(self, cluster, force=False):
        # print(f">> removing cluster: {cluster.name}")
        if force:
            nodes = self.get_clusters_nodes(cluster)
            for node in nodes[:]:
                self.nodes_to_clusters.remove_related_item(node, cluster)
                self.clusters_to_nodes.remove_related_item(cluster, node)
                
                if len(self.get_nodes_clusters(node)) == 0:
                    self.delete_node(node)

        self.clusters.remove(cluster)
        self.clusters_to_nodes.remove(cluster)
        
        # Remove from CDZ
        cluster.cdz.remove_cluster(cluster)

    def get_clusters_nodes(self, cluster, include_strengths=False):
        data = self.clusters_to_nodes.get(cluster)
        if include_strengths:
            return data['list'], data['strengths']
        return data['list']

    def get_nodes_clusters(self, node, include_strengths=False, include_all=False):
        data = self.nodes_to_clusters.get(node)
        if include_all:
            return data['list'], data['strengths'], data['position'], data['count']
        elif include_strengths:
            return data['list'], data['strengths']
        return data['list']

    def get_node_managers_nodes(self, node_manager):
        try:
            return self.node_manager_to_nodes.get(node_manager)['list']
        except KeyError:
            return []

    def adjust_node_to_cluster_strength(self, node, cluster, amount, last_encoding):
        is_node_related = self.nodes_to_clusters.is_related(node, cluster)
        
        if not is_node_related:
            self.nodes_to_clusters.add_related_item(node, cluster, amount, last_encoding)
            self.clusters_to_nodes.add_related_item(cluster, node, amount)
        else:
            self.nodes_to_clusters.increase_relationship_strength(node, cluster, amount, last_encoding)

    def adjust_cluster_to_node_strength(self, cluster, node, amount):
        self.clusters_to_nodes.increase_relationship_strength(cluster, node, amount)

    def cleanup(self):
        clusters_to_delete = []
        for cluster_data in self.clusters_to_nodes.data.values():
            cluster = cluster_data['obj']
            if cluster.is_underutilized():
                clusters_to_delete.append(cluster)
        
        for cluster in clusters_to_delete:
            self._delete_cluster(cluster, force=True)

# Global DB instance
db = Database()
