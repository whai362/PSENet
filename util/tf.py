from __future__ import print_function


try:
    import tensorflow as tf
    from tensorflow.python.ops import nn
    relu = nn.relu
    slim = tf.contrib.slim
    sigmoid = nn.sigmoid
    softmax = nn.softmax
except:
    print("tensorflow is not installed, util.tf can not be used.")

def is_gpu_available(cuda_only=True):
  """
  code from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/platform/test.py
  Returns whether TensorFlow can access a GPU.
  Args:
    cuda_only: limit the search to CUDA gpus.
  Returns:
    True iff a gpu device of the requested kind is available.
  """
  from tensorflow.python.client import device_lib as _device_lib

  if cuda_only:
    return any((x.device_type == 'GPU')
               for x in _device_lib.list_local_devices())
  else:
    return any((x.device_type == 'GPU' or x.device_type == 'SYCL')
               for x in _device_lib.list_local_devices())



def get_available_gpus(num_gpus = None):
    """
    Modified on http://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
    However, the original code will occupy all available gpu memory.
    The modified code need a parameter: num_gpus. It does nothing but return the device handler name
    It will work well on single-maching-training, but I don't know whether it will work well on a cluster.
    """
    if num_gpus == None:
        from tensorflow.python.client import device_lib as _device_lib
        local_device_protos = _device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']
    else:
        return ['/gpu:%d'%(idx) for idx in xrange(num_gpus)]

def get_latest_ckpt(path):
# tf.train.latest_checkpoint
    import util
    path = util.io.get_absolute_path(path)
    if util.io.is_dir(path):
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt is not None:
            ckpt_path = ckpt.model_checkpoint_path
        else:
            ckpt_path = None
    else:
        ckpt_path = path;
    return ckpt_path

def get_all_ckpts(path):
    ckpt = tf.train.get_checkpoint_state(path)
    all_ckpts = ckpt.all_model_checkpoint_paths
    ckpts = [str(c) for c in all_ckpts]
    return ckpts

def get_iter(ckpt):
    import util
    iter_ = int(util.str.find_all(ckpt, '.ckpt-\d+')[0].split('-')[-1])
    return iter_

def get_init_fn(checkpoint_path, train_dir, ignore_missing_vars = False,
                checkpoint_exclude_scopes = None, model_name = None, checkpoint_model_scope = None):
    """
    code from github/SSD-tensorflow/tf_utils.py
    Returns a function run by the chief worker to warm-start the training.
    Note that the init_fn is only run when initializing the model during the very
    first global step.

    checkpoint_path: the checkpoint to be restored
    train_dir: the directory where checkpoints are stored during training.
    ignore_missing_vars: if False and there are variables in the model but not in the checkpoint, an error will be raised.
    checkpoint_model_scope and model_name: if the root scope of checkpoints and the model in session is different,
            (but the sub-scopes are all the same), specify them clearly
    checkpoint_exclude_scopes: variables to be excluded when restoring from checkpoint_path.
    Returns:
      An init function run by the supervisor.
    """
    import util
    if util.str.is_none_or_empty(checkpoint_path):
        return None
    # Warn the user if a checkpoint exists in the train_dir. Then ignore.
    if tf.train.latest_checkpoint(train_dir):
        tf.logging.info(
            'Ignoring --checkpoint_path because a checkpoint already exists in %s'
            % train_dir)
        return None

    exclusions = []
    if checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in checkpoint_exclude_scopes.split(',')]

    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    # Change model scope if necessary.
    if checkpoint_model_scope is not None:
        variables_to_restore = {checkpoint_model_scope + '/' + var.op.name : var for var in variables_to_restore}
        tf.logging.info('variables_to_restore: %r'%(variables_to_restore))
    checkpoint_path = get_latest_ckpt(checkpoint_path)
    tf.logging.info('Fine-tuning from %s. Ignoring missing vars: %s' % (checkpoint_path, ignore_missing_vars))
    print ('checkpoint_path', checkpoint_path)
    return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=ignore_missing_vars)


def get_variables_to_train(flags = None):
    """code from github/SSD-tensorflow/tf_utils.py
    Returns a list of variables to train.

    Returns:
      A list of variables to train by the optimizer.
    """
    if flags is None or flags.trainable_scopes is None:
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in flags.trainable_scopes.split(',')]

    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train

def Print(tensor, data, msg = '', file = None, mode = 'w'):
    from tensorflow.python.ops import control_flow_ops
    import util
    def np_print(*args):
        if util.str.contains(msg, '%'):
            message = msg%tuple(args)
        else:
            message = msg + ' %'*len(args)%tuple(args)
        if file is not None:
            file_path = util.io.get_absolute_path(file)
            print('writting message to file(%s):'%(file_path), message)
            with open(file_path, mode) as f:
                print(message, file = f)
        else:
            print(message)
    return control_flow_ops.with_dependencies([tf.py_func(np_print, data, [])], tensor)

def get_variable_names_in_checkpoint(path, return_shapes = False, return_reader = False):
    """
    Args:
        path: the path to training directory containing checkpoints,
            or path to checkpoint
    Return:
        a list of variable names in the checkpoint
    """
    import util
    ckpt = get_latest_ckpt(path)
    ckpt_reader = tf.train.NewCheckpointReader(ckpt)
    ckpt_vars = ckpt_reader.get_variable_to_shape_map()
    names = [var for var in ckpt_vars]
    if return_shapes:
        return names, ckpt_vars
    def get(name):
        return ckpt_reader.get_tensor(name)
    if return_reader:
        return names, get
    return names



def min_area_rect(xs, ys):
    import util
    rects = tf.py_func(util.img.min_area_rect, [xs, ys], xs.dtype)
    rects.set_shape([None, 5])
    return rects


def gpu_config(config = None, allow_growth = None, gpu_memory_fraction = None):
    if config is None:
        config = tf.ConfigProto()

    if allow_growth is not None:
        config.gpu_options.allow_growth = allow_growth

    if gpu_memory_fraction is not None:
        config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction

    return config

def wait_for_checkpoint(path):
    from tensorflow.contrib.training.python.training import evaluation
    return evaluation.checkpoints_iterator(path)
    
def focal_loss(labels, logits, gamma = 2.0, alpha = 0.75, normalize = True):
    labels = tf.where(labels > 0, tf.ones_like(labels), tf.zeros_like(labels))
    labels = tf.cast(labels, tf.float32)
    probs = tf.sigmoid(logits)
    CE = tf.nn.sigmoid_cross_entropy_with_logits(labels = labels, logits = logits)

    alpha_t = tf.ones_like(logits) * alpha
    alpha_t = tf.where(labels > 0, alpha_t, 1.0 - alpha_t)
    probs_t = tf.where(labels > 0, probs, 1.0 - probs)

    focal_matrix = alpha_t * tf.pow((1.0 - probs_t), gamma)
    fl = focal_matrix * CE

    fl = tf.reduce_sum(fl)
    if normalize:
        #n_pos = tf.reduce_sum(labels)
        #fl = fl / tf.cast(n_pos, tf.float32)
        total_weights = tf.stop_gradient(tf.reduce_sum(focal_matrix))
        fl = fl / total_weights
    return fl


def focal_loss_layer_initializer(sigma = 0.01, pi = 0.01):
    import numpy as np
    b0 = - np.log((1 - pi) / pi)
    return tf.random_normal_initializer(stddev = sigma), \
            tf.constant_initializer(b0)


def sum_gradients(clone_grads, do_summary = False):                        
    averaged_grads = []
    for grad_and_vars in zip(*clone_grads):
        grads = []
        var = grad_and_vars[0][1]
        try:
            for g, v in grad_and_vars:
                assert v == var
                grads.append(g)
            grad = tf.add_n(grads, name = v.op.name + '_summed_gradients')
        except:
            import pdb
            pdb.set_trace()
        
        averaged_grads.append((grad, v))
        
        if do_summary:
            tf.summary.histogram("variables_and_gradients_" + grad.op.name, grad)
            tf.summary.histogram("variables_and_gradients_" + v.op.name, v)
            tf.summary.scalar("variables_and_gradients_" + grad.op.name+\
                  '_mean/var_mean', tf.reduce_mean(grad)/tf.reduce_mean(var))
            tf.summary.scalar("variables_and_gradients_" + v.op.name+'_mean',tf.reduce_mean(var))
    return averaged_grads

def get_update_op():
    """
    Extremely important for BatchNorm
    """
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
        return tf.group(*update_ops)
    return None
