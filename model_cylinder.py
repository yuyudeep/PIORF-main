import sonnet as snt
import tensorflow as tf

from core_model_mgn import EdgeSet, MultiGraph
from util import NodeType, triangles_to_edges, check_unique, pairwise_dist
from normalization import Normalizer


class CylinderFlow(snt.Module):
    def __init__(self, learned_model, name='Model'):
        super(CylinderFlow, self).__init__(name=name)
        self._learned_model = learned_model

        self._output_normalizer = Normalizer(size=2, name='output_normalizer')
        self._output_presuure_normalizer = Normalizer(size=1, name='output_presuure_normalizer')
        self._node_normalizer = Normalizer(size=2 + NodeType.SIZE, name='node_normalizer')
        self._edge_normalizer = Normalizer(size=3, name='edge_normalizer')
    
    def _build_graph(self, inputs, is_training):
        node_type = tf.one_hot(inputs['node_type'][:, 0], NodeType.SIZE)
        node_features = tf.concat([inputs['velocity'], node_type], axis=-1)

        """ Edge Features : Mesh """
        s, r = triangles_to_edges(inputs['cells'], True)
        senders, receivers = tf.concat([s, r], axis=0), tf.concat([r, s], axis=0)
        
        relative_mesh_pos = tf.gather(inputs['mesh_pos'], senders) - tf.gather(inputs['mesh_pos'], receivers)
        
        edge_features = tf.concat([
            relative_mesh_pos, tf.norm(relative_mesh_pos, axis=-1, keepdims=True)
            ], axis=-1)

        edges_mesh = EdgeSet(
            name='mesh_edges',
            features=self._edge_normalizer(edge_features, is_training),
            receivers=receivers,
            senders=senders)
        
        return MultiGraph(
            node_features=self._node_normalizer(node_features, is_training), 
            edge_sets=[edges_mesh]
            )
        
    def __call__(self, inputs):
        graph = self._build_graph(inputs, is_training=False)
        per_node_network_output = self._learned_model(graph)
        return self._update(inputs, per_node_network_output)

    def loss(self, inputs):
        graph = self._build_graph(inputs, is_training=True)
        network_output = self._learned_model(graph)

        cur_velocity = inputs['velocity']
        target_velocity = inputs['target|velocity']
        target_velocity_change = target_velocity - cur_velocity        
        target_normalized = self._output_normalizer(target_velocity_change)

        target_pressure = inputs['target|pressure']
        target_pressure_normalized = self._output_presuure_normalizer(target_pressure)    

        # build loss
        node_type = inputs['node_type'][:, 0]
        mask = tf.logical_or(tf.equal(node_type, NodeType.NORMAL), tf.equal(node_type, NodeType.OUTFLOW))

        error_vel = tf.reduce_sum((target_normalized - network_output[:, :2])**2, axis=1)
        error_pre = tf.reduce_sum((target_pressure_normalized - network_output[:, 2:3])**2, axis=1)

        loss_vel = tf.reduce_mean(error_vel[mask])
        loss_pre = tf.reduce_mean(error_pre)

        return loss_vel + loss_pre

    def _update(self, inputs, per_node_network_output):
        velocity_update = self._output_normalizer.inverse(per_node_network_output[:, :2])
        pred_pressure = self._output_presuure_normalizer.inverse(per_node_network_output[:, 2:3])
        cur_velocity = inputs['velocity']
        return cur_velocity + velocity_update, pred_pressure


class CylinderFlowRewire(snt.Module):
    def __init__(self, learned_model, rewire_name, name='Model'):
        super(CylinderFlowRewire, self).__init__(name=name)
        self._learned_model = learned_model

        self._output_normalizer = Normalizer(size=2, name='output_normalizer')
        self._output_presuure_normalizer = Normalizer(size=1, name='output_presuure_normalizer')
        self._node_normalizer = Normalizer(size=2 + NodeType.SIZE, name='node_normalizer')
        self._edge_normalizer = Normalizer(size=3, name='edge_normalizer')
        self.rewire_name = rewire_name

    def _build_graph(self, inputs, is_training):
        node_type = tf.one_hot(inputs['node_type'][:, 0], NodeType.SIZE)
        node_features = tf.concat([inputs['velocity'], node_type], axis=-1)
        
        """ Edge Features : Mesh """
        senders = tf.squeeze(inputs[f'senders{self.rewire_name}'])
        receivers = tf.squeeze(inputs[f'receivers{self.rewire_name}'])
        senders.set_shape([None])
        receivers.set_shape([None])

        relative_mesh_pos = tf.gather(inputs['mesh_pos'], senders) - tf.gather(inputs['mesh_pos'], receivers)
        
        edge_features = tf.concat([
            relative_mesh_pos, tf.norm(relative_mesh_pos, axis=-1, keepdims=True)
            ], axis=-1)

        edges_mesh = EdgeSet(
            name='mesh_edges',
            features=self._edge_normalizer(edge_features, is_training),
            receivers=receivers,
            senders=senders)
        
        return MultiGraph(node_features=self._node_normalizer(node_features, is_training), edge_sets=[edges_mesh])
        
    def __call__(self, inputs):
        graph = self._build_graph(inputs, is_training=False)
        per_node_network_output = self._learned_model(graph)
        return self._update(inputs, per_node_network_output)

    def loss(self, inputs):
        graph = self._build_graph(inputs, is_training=True)
        network_output = self._learned_model(graph)

        cur_velocity = inputs['velocity']
        target_velocity = inputs['target|velocity']
        target_velocity_change = target_velocity - cur_velocity        
        target_normalized = self._output_normalizer(target_velocity_change)

        target_pressure = inputs['target|pressure']
        target_pressure_normalized = self._output_presuure_normalizer(target_pressure)    

        # build loss
        node_type = inputs['node_type'][:, 0]
        mask = tf.logical_or(tf.equal(node_type, NodeType.NORMAL), tf.equal(node_type, NodeType.OUTFLOW))

        error_vel = tf.reduce_sum((target_normalized - network_output[:, :2])**2, axis=1)
        error_pre = tf.reduce_sum((target_pressure_normalized - network_output[:, 2:3])**2, axis=1)

        loss_vel = tf.reduce_mean(error_vel[mask])
        loss_pre = tf.reduce_mean(error_pre)

        return loss_vel + loss_pre
    
    def _update(self, inputs, per_node_network_output):
        velocity_update = self._output_normalizer.inverse(per_node_network_output[:, :2])
        pred_pressure = self._output_presuure_normalizer.inverse(per_node_network_output[:, 2:3])
        cur_velocity = inputs['velocity']
        return cur_velocity + velocity_update, pred_pressure


class CylinderFlowPIRF(snt.Module):
    def __init__(self, learned_model, name='Model'):
        super(CylinderFlowPIRF, self).__init__(name=name)
        self._learned_model = learned_model

        self._output_normalizer = Normalizer(size=2, name='output_normalizer')
        self._output_presuure_normalizer = Normalizer(size=1, name='output_presuure_normalizer')
        self._node_normalizer = Normalizer(size=2 + NodeType.SIZE, name='node_normalizer')
        self._edge_normalizer = Normalizer(size=3, name='edge_normalizer')
        self.pool = snt.Linear(1)

        self.pooling_ratio = 3
        self.mode_add = True
        self.mode_del = False

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.int32), 
        tf.TensorSpec(shape=[None, 2], dtype=tf.int32), 
        tf.TensorSpec(shape=[None, 1], dtype=tf.float32)])
    def get_del(self, nodes, edges, feat):
        m = tf.shape(edges)
        edges_flat = tf.reshape(edges, [-1])
        
        masks = tf.math.reduce_any(tf.reshape(nodes[:, None]==edges_flat[None, :], [-1, m[0], 2]), 2)
        mask_shape = tf.shape(masks)
        edges_bc = tf.broadcast_to(edges, [mask_shape[0], mask_shape[1], 2])

        sub_edges = tf.ragged.boolean_mask(edges_bc, masks)
        sens = tf.gather(feat, sub_edges[:, :, 0])
        revs = tf.gather(feat, sub_edges[:, :, 1])
        idxs = tf.multiply(sens, revs)

        idxs = tf.map_fn(tf.math.argmin, idxs, fn_output_signature=tf.int64)
        
        edge = tf.gather(sub_edges, idxs, batch_dims=1)
        edge = tf.squeeze(edge, 1)

        dels = edge[:, 0]
        delr = edge[:, 1]
        return dels, delr
    
    def _build_graph(self, inputs, is_training):
        node_type = tf.one_hot(inputs['node_type'][:, 0], NodeType.SIZE)
        node_features = tf.concat([inputs['velocity'], node_type], axis=-1)

        """ Edge Features : Mesh """
        s, r = triangles_to_edges(inputs['cells'], True)
        senders, receivers = tf.concat([s, r], axis=0), tf.concat([r, s], axis=0)
        
        N = tf.cast(tf.shape(inputs['mesh_pos'])[0], tf.float32) * tf.constant(self.pooling_ratio /100, tf.float32)
        N = tf.cast(N, tf.int32)
        
        feat = self.pool(inputs['velocity'])

        senders_add = tf.squeeze(inputs[f'senders_pirf'])
        senders_add.set_shape([None])


        # adding
        if self.mode_add:
            senders_add = senders_add[:N]
            feat_top = tf.gather(feat, senders_add)
            cdist = pairwise_dist(feat, feat_top)
            receivers_add = tf.argmax(cdist, axis=0)
            receivers_add = tf.cast(receivers_add, tf.int32)
            senders_add, receivers_add = tf.concat([senders_add, receivers_add], axis=0), tf.concat([receivers_add, senders_add], axis=0)

            senders = tf.concat([senders, senders_add], axis=0)
            receivers = tf.concat([receivers, receivers_add], axis=0)

        # del
        if self.mode_del:
            senders_del = senders_add[-1*N:]
            edges = tf.stack([senders, receivers], 1)

            senders_del, receivers_del = self.get_del(senders_del, edges, feat)
            senders_del, receivers_del = tf.concat([senders_del, receivers_del], axis=0), tf.concat([receivers_del, senders_del], axis=0)

            edges = tf.expand_dims(tf.bitcast(tf.stack([senders, receivers], axis=1), tf.int64), 0)
            edges = tf.expand_dims(edges, 0)

            edges_del = tf.expand_dims(tf.bitcast(tf.stack([senders_del, receivers_del], axis=1), tf.int64), 0)
            edges_del = tf.expand_dims(edges_del, 0)

            edges = tf.sets.difference(edges, edges_del, aminusb=True, validate_indices=True).values
            edges = tf.bitcast(edges, tf.int32)
            senders, receivers = tf.unstack(edges, axis=1)

        senders, receivers = check_unique(senders, receivers)
        
        relative_mesh_pos = tf.gather(inputs['mesh_pos'], senders) - tf.gather(inputs['mesh_pos'], receivers)
        
        edge_features = tf.concat([
            relative_mesh_pos, tf.norm(relative_mesh_pos, axis=-1, keepdims=True)
            ], axis=-1)

        edges_mesh = EdgeSet(
            name='mesh_edges',
            features=self._edge_normalizer(edge_features, is_training),
            receivers=receivers,
            senders=senders)
        
        return MultiGraph(
            node_features=self._node_normalizer(node_features, is_training), 
            edge_sets=[edges_mesh]
            )
        
    def __call__(self, inputs):
        graph = self._build_graph(inputs, is_training=False)
        per_node_network_output = self._learned_model(graph)
        return self._update(inputs, per_node_network_output)

    def loss(self, inputs):
        graph = self._build_graph(inputs, is_training=True)
        network_output = self._learned_model(graph)

        cur_velocity = inputs['velocity']
        target_velocity = inputs['target|velocity']
        target_velocity_change = target_velocity - cur_velocity        
        target_normalized = self._output_normalizer(target_velocity_change)

        target_pressure = inputs['target|pressure']
        target_pressure_normalized = self._output_presuure_normalizer(target_pressure)    

        # build loss
        node_type = inputs['node_type'][:, 0]
        mask = tf.logical_or(tf.equal(node_type, NodeType.NORMAL), tf.equal(node_type, NodeType.OUTFLOW))

        error_vel = tf.reduce_sum((target_normalized - network_output[:, :2])**2, axis=1)
        error_pre = tf.reduce_sum((target_pressure_normalized - network_output[:, 2:3])**2, axis=1)

        loss_vel = tf.reduce_mean(error_vel[mask])
        loss_pre = tf.reduce_mean(error_pre)

        return loss_vel + loss_pre

    
    def _update(self, inputs, per_node_network_output):
        velocity_update = self._output_normalizer.inverse(per_node_network_output[:, :2])
        pred_pressure = self._output_presuure_normalizer.inverse(per_node_network_output[:, 2:3])
        cur_velocity = inputs['velocity']
        return cur_velocity + velocity_update, pred_pressure
