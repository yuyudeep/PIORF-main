import tensorflow as tf
from util import NodeType

def evaluate_cylinder(model, inputs):
    def _rollout(model, initial_state, num_steps):
        mask = tf.logical_or(tf.equal(initial_state['node_type'], NodeType.NORMAL), tf.equal(initial_state['node_type'], NodeType.OUTFLOW))
        
        def step_fn(step, velocity, pressure, trajectory, trajectory_pressure):
            pred_vel, next_pressure = model({**initial_state, 'velocity': velocity, 'pressure':pressure})

            next_velocity = tf.where(mask, pred_vel, velocity)

            trajectory = trajectory.write(step, velocity)
            trajectory_pressure = trajectory_pressure.write(step, pressure)

            return step+1, next_velocity, next_pressure, trajectory, trajectory_pressure

        _, _, _, output, output_pressure = tf.while_loop(
        cond=lambda step, cur, cur_pre, traj, traj_pressure: tf.less(step, num_steps),
        body=step_fn,
        loop_vars=(0, initial_state['velocity'], initial_state['pressure'], tf.TensorArray(tf.float32, num_steps), tf.TensorArray(tf.float32, num_steps)), parallel_iterations=1)
        return output.stack(), output_pressure.stack()

    initial_state = {k: v[0] for k, v in inputs.items()}
    num_steps = inputs['mesh_pos'].shape[0]
    pred_vel, pred_pressure = _rollout(model, initial_state, num_steps)
    
    error_rmse = tf.sqrt(tf.reduce_mean(tf.reduce_sum((pred_vel - inputs['velocity'])**2, axis=-1), -1))
    error_pressure_rmse = tf.sqrt(tf.reduce_mean(tf.reduce_sum((pred_pressure - inputs['pressure'])**2, axis=-1), -1))

    scalars = {'%d' % horizon: tf.reduce_mean(error_rmse[1:horizon+1]).numpy() * 1E3 for horizon in [1, error_rmse.shape[0]]}
    scalars_pressure = {'s%d' % horizon: tf.reduce_mean(error_pressure_rmse[1:horizon+1]).numpy() * 1E3 for horizon in [1, error_pressure_rmse.shape[0]]}
    scalars.update(scalars_pressure)

    traj_ops = {
        'faces': inputs['cells'],
        'mesh_pos': inputs['mesh_pos'],
        'gt_velocity': inputs['velocity'],
        'gt_pressure': inputs['pressure'],
        'node_type': inputs['node_type'],
        'pred_velocity': pred_vel,
        'pred_pressure': pred_pressure
    }
    return scalars, traj_ops
