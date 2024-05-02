import tensorflow as tf
import collections
import sonnet as snt


EdgeSet = collections.namedtuple('EdgeSet', ['name', 'features', 'senders', 'receivers'])
MultiGraph = collections.namedtuple('Graph', ['node_features', 'edge_sets'])


class MLP(snt.Module):
    def __init__(self, num_layers, latent_size, output_size, layer_norm=True, name=None):
        super(MLP, self).__init__(name=name)
        widths = [latent_size] * num_layers + [output_size]

        self.network = snt.nets.MLP(widths, activate_final=False)

        if layer_norm:
            self.network = snt.Sequential([self.network, snt.LayerNorm(axis=-1, create_scale=True, create_offset=True)])
        

    def __call__(self, inputs):
        return self.network(inputs)


class Encoder(snt.Module):
    def __init__(self, name='encoder'):
        super(Encoder, self).__init__(name=name)

        num_layers = 2
        latent_size = 128

        self.node_latents = MLP(num_layers, latent_size, latent_size, name=f'{name}_node')
        self.edge_latents = [MLP(num_layers, latent_size, latent_size, name=f'{name}_edge_mesh')]

    def __call__(self, graph):
        node_latents = self.node_latents(graph.node_features)

        new_edges_sets = []
        for edge_set, network in zip(graph.edge_sets, self.edge_latents):
            latent = network(edge_set.features)
            new_edges_sets.append(edge_set._replace(features=latent))
        return MultiGraph(node_latents, new_edges_sets)


class GraphNetBlock(snt.Module):
    def __init__(self, name='GraphNetBlock'):
        super(GraphNetBlock, self).__init__(name=name)

        num_layers = 2
        latent_size = 128

        self.node_network = MLP(num_layers, latent_size, latent_size, name=f'{name}_node')
        self.edge_networks = [MLP(num_layers, latent_size, latent_size, name=f'{name}_edge_mesh')]

    def update_edge_features(self, node_features, edge_set):
        sender_features = tf.gather(node_features, edge_set.senders)
        receiver_features = tf.gather(node_features, edge_set.receivers)

        features = [sender_features, receiver_features, edge_set.features]
        return tf.concat(features, axis=-1)

    def update_node_features(self, node_features, edge_sets):
        num_nodes = tf.shape(node_features)[0]
        features = [node_features]
        for edge_set in edge_sets:
            features.append(tf.math.unsorted_segment_sum(edge_set.features, edge_set.receivers, num_nodes))
        return tf.concat(features, axis=-1)


    def __call__(self, graph):
        new_edge_sets = []

        for edge_set, network in zip(graph.edge_sets, self.edge_networks):
            updated_features = network(self.update_edge_features(graph.node_features, edge_set))
            new_edge_sets.append(edge_set._replace(features=updated_features))

        node_features = self.update_node_features(graph.node_features, new_edge_sets)
        new_node_features = self.node_network(node_features)

        # add residual connections
        new_node_features += graph.node_features
        new_edge_sets = [es._replace(features=es.features + old_es.features) for es, old_es in zip(new_edge_sets, graph.edge_sets)] # No

        return MultiGraph(new_node_features, new_edge_sets)


class MGN(snt.Module):
    def __init__(self, output_size, name='MGN'):
        super(MGN, self).__init__(name=name)

        """ Networks """
        self.encoder = Encoder(name='encoder')
        self.graphnetblocks = [GraphNetBlock(f'GraphNetBlock_{i}') for i in range(15)]
        self.decoder = MLP(2, 128, output_size, layer_norm=False, name='decoder')
        
    def __call__(self, graph):
        latent_graph = self.encoder(graph)
        
        for network in self.graphnetblocks:
            latent_graph = network(latent_graph)
        return self.decoder(latent_graph.node_features)
