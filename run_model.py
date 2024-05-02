import tensorflow as tf
import numpy as np
import os
import dataset
import random
import pickle

from pathlib import Path
from util import get_check_point_num
from termcolor import colored
from absl import app
from absl import flags
from absl import logging

from core_model_mgn import MGN

from model_cylinder import CylinderFlow, CylinderFlowRewire, CylinderFlowPIRF
from rollout import evaluate_cylinder



print(colored(f'tensorflow version : {tf.__version__}', 'red'))
print(colored(f'GPUs Available : {len(tf.config.experimental.list_physical_devices("GPU"))}', 'red'))

logger_tf = tf.get_logger()
logger_tf.setLevel('ERROR')
logger_tf.propagate = False

gpus = tf.config.list_physical_devices('GPU')

FLAGS = flags.FLAGS
flags.DEFINE_enum('mode', 'train', ['train', 'eval'], 'Train model, or run evaluation.')
flags.DEFINE_enum('rewire', None, ['DIGL', 'SDRF', 'FoSR', 'BORF', 'PIRF'], 'rewire')
flags.DEFINE_enum('model', "MGN", ['MGN'], 'core models')
flags.DEFINE_integer('num_rollouts', 1, 'No. of rollouts')
flags.DEFINE_integer('seed', 10, 'No. of random seed')


dataset_dir =  f"data/cylinder_flow"

for i in range(len(gpus)):
	tf.config.experimental.set_memory_growth(gpus[i], True)


def learner(model, task):   
    ds = dataset.load_dataset(dataset_dir, 'train', task['rewire_name'])
    ds = dataset.add_targets(ds, ['velocity', 'pressure', 'density'], add_history=False)
    ds = dataset.split_and_preprocess(ds, FLAGS.seed)
    
    @tf.function(input_signature=[ds.element_spec])
    def train_step(inputs):
        with tf.GradientTape() as tape:
            loss = model.loss(inputs)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss
            
    global_step = tf.Variable(0, name='global_step', trainable=False)

    ckpt = tf.train.Checkpoint(step=global_step, net=model)

    manager = tf.train.CheckpointManager(checkpoint=ckpt, directory=task['dir_checkpoint'], max_to_keep=50)
    
    ckpt.restore(manager.latest_checkpoint)
    
    lr_schedule = tf.compat.v1.train.exponential_decay(learning_rate=1e-4, global_step=global_step, decay_steps=5000000, decay_rate=0.1)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    ds = iter(ds)
    
    print(optimizer._decayed_lr(tf.float32).numpy())
    print('global step', global_step.numpy())

    losses = 0

    """ Training """
    counter = 0
    total_steps = 10000000
    epoch_steps = 598000

    for step in range(int(global_step), total_steps, 1):
        inputs = ds.get_next()
        if step < int(1000):
            model._build_graph(inputs, True)
        else:
            loss = train_step(inputs)
            losses += loss
            counter += 1

            if counter != 1 and step % epoch_steps == 0:
                manager.save(checkpoint_number=int(global_step))
            
            if counter != 1 and step % int(epoch_steps/200) == 0:
                with open(os.path.join(task['dir_log'], 'train_MSE.txt'), 'a') as file:
                    file.write(f'{step} {losses/counter:.9f}\n')

        global_step.assign_add(1)

    manager.save(checkpoint_number=int(global_step))


def evaluator(model, task):
    ds = dataset.load_dataset(dataset_dir, 'test', task['rewire_name'])
    ds = dataset.add_targets(ds, ['velocity', 'pressure', 'density'], add_history=False)
    ds = iter(ds)

    trajectories = []
    scalars = []

    optimizer = tf.keras.optimizers.Adam()
    ckpt = tf.train.Checkpoint(optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(checkpoint=ckpt, directory=task['dir_checkpoint'], max_to_keep=None)
    ckpt.restore(manager.latest_checkpoint)
    checkpoint_num = get_check_point_num(os.path.join(task['dir_checkpoint'], 'checkpoint'))

    print(colored(checkpoint_num, 'red'))
    counter = 0
    for traj_idx in range(FLAGS.num_rollouts):
        inputs = ds.get_next()
        scalar_data, traj_data = evaluate_cylinder(model, inputs)
        trajectories.append(traj_data)
        scalars.append(scalar_data)
        print(traj_idx, scalar_data)
        counter += 1
        del traj_data
        del inputs

    with open(os.path.join(task['dir_log'], 'test_RMSE.txt'), 'a') as file:
        txt = ''
        for key in scalars[0]:
            print('%s: %g', key, np.mean([x[key] for x in scalars]))
            txt += f' {key} {np.mean([x[key] for x in scalars])}'
        file.write(f'{checkpoint_num} {txt}\n')
    
    with open(os.path.join(task['dir_rollout'], f'{checkpoint_num}.pkl'), 'wb') as fp:
        pickle.dump(trajectories, fp)


def main(argv):
    del argv

    tf.compat.v1.enable_resource_variables()
    tf.config.run_functions_eagerly(False)
    
    name = f"{FLAGS.model}_{FLAGS.rewire}"


    """ Create base directory """
    dir_checkpoint = os.path.join(f"workspace/{name}/check")
    dir_rollout = os.path.join(f"workspace/{name}/rollout")
    dir_log = os.path.join(f"workspace/{name}/log")
    
    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
    Path(dir_rollout).mkdir(parents=True, exist_ok=True)
    Path(dir_log).mkdir(parents=True, exist_ok=True)

    task = {}
    task['dir_checkpoint'] = dir_checkpoint
    task['dir_rollout'] = dir_rollout
    task['dir_log'] = dir_log

    """ Fix seed """
    tf.keras.utils.set_random_seed(FLAGS.seed)
    tf.config.experimental.enable_op_determinism()
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    tf.random.set_seed(FLAGS.seed)
    tf.compat.v1.set_random_seed(FLAGS.seed)

    if FLAGS.rewire == None:
        rewire_name = None
        model = CylinderFlow(MGN(output_size=3))
    elif FLAGS.rewire == 'DIGL':
        rewire_name = '_digl'
        model = CylinderFlowRewire(MGN(output_size=3), rewire_name)
    elif FLAGS.rewire == 'SDRF':
        rewire_name = '_sdrf'
        model = CylinderFlowRewire(MGN(output_size=3), rewire_name)           
    elif FLAGS.rewire == 'FoSR':
        rewire_name = '_fosr'
        model = CylinderFlowRewire(MGN(output_size=3), rewire_name)  
    elif FLAGS.rewire == 'BORF':
        rewire_name = '_borf'
        model = CylinderFlowRewire(MGN(3), rewire_name)
    elif FLAGS.rewire == 'PIRF':
        rewire_name = '_pirf'
        model = CylinderFlowPIRF(MGN(3))

    task['rewire_name'] = rewire_name

    if FLAGS.mode == 'train':
        learner(model, task)

    if FLAGS.mode == 'eval':
        evaluator(model, task)


if __name__ == '__main__':
    app.run(main)
