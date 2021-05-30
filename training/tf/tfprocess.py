#!/usr/bin/env python3
#
#    This file is part of SAI, which is a fork of Leela Zero.
#    Copyright (C) 2017-2018 Gian-Carlo Pascutto
#    Copyright (C) 2018-2019 SAI Team
#
#    SAI is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    SAI is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with SAI.  If not, see <http://www.gnu.org/licenses/>.

import math
import numpy as np
import os
import tensorflow as tf
import time
import unittest

from mixprec import float32_variable_storage_getter, LossScalingOptimizer

from config import *

def weight_variable(name, shape, dtype):
    """Xavier initialization"""
    stddev = np.sqrt(2.0 / (sum(shape)))
    # Do not use a constant as the initializer, that will cause the
    # variable to be stored in wrong dtype.
    weights = tf.get_variable(
        name, shape, dtype=dtype,
        initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights)
    return weights

# Bias weights for layers not followed by BatchNorm
# We do not regularlize biases, so they are not
# added to the regularlizer collection
def bias_variable(name, shape, dtype):
    bias = tf.get_variable(name, shape, dtype=dtype,
                           initializer=tf.zeros_initializer())
    return bias


def conv2d(x, W):
    return tf.nn.conv2d(x, W, data_format='NCHW',
                        strides=[1, 1, 1, 1], padding='SAME')

# Restore session from checkpoint. It silently ignore mis-matches
# between the checkpoint and the graph. Specifically
# 1. values in the checkpoint for which there is no corresponding variable.
# 2. variables in the graph for which there is no specified value in the
#    checkpoint.
# 3. values where the checkpoint shape differs from the variable shape.
# (variables without a value in the checkpoint are left at their default
# initialized value)
def optimistic_restore(session, save_file, graph=tf.get_default_graph()):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted(
        [(var.name, var.name.split(':')[0]) for var in tf.global_variables()
         if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    for var_name, saved_var_name in var_names:
        curr_var = graph.get_tensor_by_name(var_name)
        var_shape = curr_var.get_shape().as_list()
        if var_shape == saved_shapes[saved_var_name]:
            restore_vars.append(curr_var)
    opt_saver = tf.train.Saver(restore_vars)
    opt_saver.restore(session, save_file)

# Class holding statistics
class Stats:
    def __init__(self):
        self.s = {}
    def add(self, stat_dict):
        for (k,v) in stat_dict.items():
            if k not in self.s:
                self.s[k] = []
            self.s[k].append(v)
    def n(self, name):
        return len(self.s[name] or [])
    def mean(self, name):
        return np.mean(self.s[name] or [0])
    def stddev_mean(self, name):
        # standard deviation in the sample mean.
        return math.sqrt(
            np.var(self.s[name] or [0]) / max(0.0001, (len(self.s[name]) - 1)))
    def str(self):
        return ', '.join(
            ["{}={:g}".format(k, np.mean(v or [0])) for k,v in self.s.items()])
    def clear(self):
        self.s = {}
    def summaries(self, tags):
        return [tf.Summary.Value(
            tag=k, simple_value=self.mean(v)) for k,v in tags.items()]

# Simple timer
class Timer:
    def __init__(self):
        self.last = time.time()
    def elapsed(self):
        # Return time since last call to 'elapsed()'
        t = time.time()
        e = t - self.last
        self.last = t
        return e

class TFProcess:
    def __init__(self, resconv_blocks, residual_filters,
                 s_rate, s_minsteps, s_steps, s_maxsteps, s_maxkeep,
                 s_policyloss, s_mseloss, s_kleloss, s_axbloss, s_regloss, s_betascale, s_plc):
        # Network structure
        self.resconv_blocks = resconv_blocks
        self.residual_filters = residual_filters

        # model type: full precision (fp32) or mixed precision (fp16)
        self.model_dtype = tf.float32

        # Scale the loss to prevent gradient underflow
        self.loss_scale = 1 if self.model_dtype == tf.float32 else 128

        # Scale for axb term
        self.axb_scale = 1

        # L2 regularization parameter applied to weights.
        self.l2_scale = 1e-4

        # Set number of GPUs for training
        self.gpus_num = 1

        # For exporting
        self.weights = []

        self.train_rate = s_rate
        self.min_steps = s_minsteps
        self.train_steps = s_steps
        self.max_steps = s_maxsteps
        self.max_keep = s_maxkeep
        self.policy_loss_wt = s_policyloss
        self.mse_loss_wt = s_mseloss
        self.kle_loss_wt = s_kleloss
        self.axb_loss_wt = s_axbloss
        self.reg_loss_wt = s_regloss
        self.beta_scale = s_betascale
        self.pol_layer_calib = s_plc

        # Output weight file with averaged weights
        self.swa_enabled = False

        # Net sampling rate (e.g 2 == every 2nd network).
        self.swa_c = 1

        # Take an exponentially weighted moving average over this
        # many networks. Under the SWA assumptions, this will reduce
        # the distance to the optimal value by a factor of 1/sqrt(n)
        self.swa_max_n = 16

        # Recalculate SWA weight batchnorm means and variances
        self.swa_recalc_bn = False

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_MEM_FRACTION)
        config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)

        self.session = tf.Session(config=config)

        self.training = tf.placeholder(tf.bool)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

    def init(self, batch_size, macrobatch=1, gpus_num=None, logbase='leelalogs'):
        self.batch_size = batch_size
        self.macrobatch = macrobatch
        self.train_rate /= self.macrobatch
        self.logbase = logbase
        # Input batch placeholders
        self.planes = tf.placeholder(tf.string, name='in_planes')
        self.probs = tf.placeholder(tf.string, name='in_probs')
        self.komi = tf.placeholder(tf.string, name='in_komi')
        self.winner = tf.placeholder(tf.string, name='in_winner')
        self.alpha = tf.placeholder(tf.string, name='in_alpha')
        self.beta = tf.placeholder(tf.string, name='in_beta')

        # Mini-batches come as raw packed strings. Decode
        # into tensors to feed into network.
        planes = tf.decode_raw(self.planes, tf.uint8)
        probs = tf.decode_raw(self.probs, tf.float32)
        komi = tf.decode_raw(self.komi, tf.float32)
        winner = tf.decode_raw(self.winner, tf.float32)
        alpha = tf.decode_raw(self.alpha, tf.float32)
        beta = tf.decode_raw(self.beta, tf.float32)

        planes = tf.cast(planes, self.model_dtype)

        planes = tf.reshape(planes, (batch_size, INPUT_PLANES + 1 + INPUT_STM, BOARD_SQUARES))
        probs = tf.reshape(probs, (batch_size, BOARD_SQUARES + 1))
        komi = tf.reshape(komi, (batch_size, 1))
        winner = tf.reshape(winner, (batch_size, 1))
        alpha = tf.reshape(alpha, (batch_size, 1))
        beta = tf.reshape(beta, (batch_size, 1))

        if gpus_num is None:
            gpus_num = self.gpus_num
        self.init_net(planes, probs, komi, winner, alpha, beta, gpus_num)

    def init_net(self, planes, probs, komi, winner, alpha, beta, gpus_num):
        self.y_ = probs  # (tf.float32, [None, BOARD_SQUARE + 1])
        self.sx = tf.split(planes, gpus_num)
        self.sk = tf.split(komi, gpus_num)     # (tf.float32, [None, 1])
        self.sy_ = tf.split(probs, gpus_num)
        self.sz_ = tf.split(winner, gpus_num)
        self.su_ = tf.split(alpha, gpus_num)
        self.sv_ = tf.split(beta, gpus_num)
        self.batch_norm_count = 0
        self.reuse_var = None

        # You need to change the learning rate here if you are training
        # from a self-play training set, for example start with 0.005 instead.
        print(f'Debug: learning rate {self.train_rate}')
        opt = tf.train.MomentumOptimizer(
            learning_rate=self.train_rate, momentum=0.9, use_nesterov=True)

        opt = LossScalingOptimizer(opt, scale=self.loss_scale)

        # Construct net here.
        tower_grads = []
        tower_loss = []
        tower_policy_loss = []
        tower_mse_loss = []
        tower_kle_loss = []
        tower_axb_loss = []
        tower_reg_term = []
        tower_y_conv = []
        with tf.variable_scope("fp32_storage",
                               # this forces trainable variables to be stored as fp32
                               custom_getter=float32_variable_storage_getter):
            for i in range(gpus_num):
                with tf.device("/gpu:%d" % i):
                    with tf.name_scope("tower_%d" % i):
                        loss, policy_loss, mse_loss, kle_loss, axb_loss, reg_term, y_conv = self.tower_loss(
                            self.sx[i], self.sk[i], self.sy_[i], self.sz_[i], self.su_[i], self.sv_[i])

                        # Reset batchnorm key to 0.
                        self.reset_batchnorm_key()

                        tf.get_variable_scope().reuse_variables()
                        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                            grads = opt.compute_gradients(loss)

                        tower_grads.append(grads)
                        tower_loss.append(loss)
                        tower_policy_loss.append(policy_loss)
                        tower_mse_loss.append(mse_loss)
                        tower_kle_loss.append(kle_loss)
                        tower_axb_loss.append(axb_loss)
                        tower_reg_term.append(reg_term)
                        tower_y_conv.append(y_conv)

        # Average gradients from different GPUs
        self.loss = tf.reduce_mean(tower_loss)
        self.policy_loss = tf.reduce_mean(tower_policy_loss)
        self.mse_loss = tf.reduce_mean(tower_mse_loss)
        self.kle_loss = tf.reduce_mean(tower_kle_loss)
        self.axb_loss = tf.reduce_mean(tower_axb_loss)
        self.reg_term = tf.reduce_mean(tower_reg_term)
        self.y_conv = tf.concat(tower_y_conv, axis=0)
        self.mean_grads = self.average_gradients(tower_grads)

        # Do swa after we contruct the net
        if self.swa_enabled is True:
            # Count of networks accumulated into SWA
            self.swa_count = tf.Variable(0., name='swa_count', trainable=False)
            # Count of networks to skip
            self.swa_skip = tf.Variable(self.swa_c, name='swa_skip',
                trainable=False)
            # Build the SWA variables and accumulators
            accum=[]
            load=[]
            n = self.swa_count
            for w in self.weights:
                name = w.name.split(':')[0]
                var = tf.Variable(
                    tf.zeros(shape=w.shape), name='swa/'+name, trainable=False)
                accum.append(
                    tf.assign(var, var * (n / (n + 1.)) + w * (1. / (n + 1.))))
                load.append(tf.assign(w, var))
            with tf.control_dependencies(accum):
                self.swa_accum_op = tf.assign_add(n, 1.)
            self.swa_load_op = tf.group(*load)

        # Accumulate gradients
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        total_grad=[]
        grad_ops=[]
        clear_var=[]
        self.grad_op_real = self.mean_grads
        for (g, v) in self.grad_op_real:
            if g is None:
                total_grad.append((g,v))
            name = v.name.split(':')[0]
            gsum = tf.get_variable(name='gsum/'+name,
                                   shape=g.shape,
                                   trainable=False,
                                   initializer=tf.zeros_initializer)
            total_grad.append((gsum, v))
            grad_ops.append(tf.assign_add(gsum, g))
            clear_var.append(gsum)
        # Op to compute gradients and add to running total in 'gsum/'
        self.grad_op = tf.group(*grad_ops)

        # Op to apply accmulated gradients
        self.train_op = opt.apply_gradients(total_grad)

        zero_ops = []
        for g in clear_var:
            zero_ops.append(
                tf.assign(g, tf.zeros(shape=g.shape, dtype=g.dtype)))
        # Op to clear accumulated gradients
        self.clear_op = tf.group(*zero_ops)

        # Op to increment global step counter
        self.step_op = tf.assign_add(self.global_step, 1)

        correct_prediction = \
            tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        self.accuracy = tf.reduce_mean(correct_prediction)

        # Summary part
        self.test_writer = tf.summary.FileWriter(
            os.path.join(os.getcwd(),
                         self.logbase + "/test"), self.session.graph)
        self.trset_test_writer = tf.summary.FileWriter(
            os.path.join(os.getcwd(),
                         self.logbase + "/trset_test"), self.session.graph)
        self.train_writer = tf.summary.FileWriter(
            os.path.join(os.getcwd(),
                         self.logbase + "/train"), self.session.graph)

        # Build checkpoint saver
        self.saver = tf.train.Saver(max_to_keep = self.max_keep)

        # Initialize all variables
        self.session.run(tf.global_variables_initializer())

    def average_gradients(self, tower_grads):
        # Average gradients from different GPUs
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, dim=0)
                grads.append(expanded_g)

            grad = tf.concat(grads, axis=0)
            grad = tf.reduce_mean(grad, reduction_indices=0)

            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def get_halflogits(self, alpha, halfbeta, komi):
        x = alpha + komi
        if VBE_NUM == 1:
            return halfbeta * x
        else:
            return tf.reduce_sum(halfbeta * np.array([-1,1], ndmin=2) \
                    * tf.nn.relu(x * np.array([-1,1], ndmin=2)), axis=-1, keepdims=True)

    def tower_loss(self, x, komi, policy_dt, outcome_dt, alpha_dt, beta_dt):
        # policylogits_nn, alpha_nn, halfbeta_nn = self.construct_net(x, komi)  # policy, win-log-odds, alpha, beta
        policylogits_nn, alpha_nn, halfbeta_nn = self.construct_net(x, komi)  # policy, alpha, beta/2

        # Cast the nn result back to fp32 to avoid loss overflow/underflow
        if self.model_dtype != tf.float32:
            policylogits_nn = tf.cast(policylogits_nn, tf.float32)
            alpha_nn = tf.cast(alpha_nn, tf.float32)
            halfbeta_nn = tf.cast(halfbeta_nn, tf.float32)

        # Calculate loss on policy head
        cross_entropy = \
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=policy_dt,
                                                       logits=policylogits_nn)
        visits_base_entropy = -tf.reduce_mean(tf.reduce_sum(tf.math.xlogy(policy_dt, policy_dt), axis=-1))
        policy_loss = tf.reduce_mean(cross_entropy) - visits_base_entropy

        loss = self.policy_loss_wt * policy_loss # first term, always present


        # Regularizer
        reg_variables = tf.get_collection(tf.GraphKeys.WEIGHTS)
        reg_term = self.l2_scale * tf.add_n(
            [tf.cast(tf.nn.l2_loss(v), tf.float32) for v in reg_variables])

        if self.reg_loss_wt > 0.0:
            loss += self.reg_loss_wt * reg_term

        # Losses on value head: MSE, KLE, AXB

        # MSE (quadratic) loss on value

        # Every data point is split in two. If the komi is not integer this has no effect
        # as 'low' and 'high' are always identical to the original values.
        # If the komi is integer, and the game is not jigo, 'low' and 'high' are identical
        # but the komi is corrected by half point in the direction in which the result
        # would eventually change, to improve the sharpness of the data point itself.
        # If the komi is integer, and the game is jigo, 'low' and 'high' are different
        # data points, with komi shifted by plus/minus 0.5 and outcome win/lose, again, to
        # improve the sharpness of the data point.
        # This enhancement improves beta in end game positions greatly.
        
        eps = 0.05
        k = 0.25
        komi_low = tf.math.floor(komi - eps - k * outcome_dt) + 0.5
        komi_high = tf.math.floor(komi + eps - k * outcome_dt) + 0.5
        outcome_dt_low = tf.math.sign(outcome_dt - k)
        outcome_dt_high = tf.math.sign(outcome_dt + k)
        halflogits_nn = self.get_halflogits(alpha_nn, halfbeta_nn, komi)
        halflogits_nn_low = self.get_halflogits(alpha_nn, halfbeta_nn, komi_low)
        halflogits_nn_high = self.get_halflogits(alpha_nn, halfbeta_nn, komi_high)
        outcome_nn = tf.nn.tanh(halflogits_nn)
        outcome_nn_low = tf.nn.tanh(halflogits_nn_low)
        outcome_nn_high = tf.nn.tanh(halflogits_nn_high)
        mse_loss = \
            0.5 * tf.reduce_mean(tf.squared_difference(outcome_dt_low, outcome_nn_low)) + \
            0.5 * tf.reduce_mean(tf.squared_difference(outcome_dt_high, outcome_nn_high))

        # winrate_conv = tf.nn.tanh(alpha_nn)
        # mse_loss = \
        #     tf.reduce_mean(tf.squared_difference(outcome_dt, winrate_conv))

        if self.mse_loss_wt > 0.0:
            loss += self.mse_loss_wt * mse_loss

        # KL loss on value
        winrate_dt_low = (outcome_dt_low + 1.0) / 2.0
        winrate_dt_high = (outcome_dt_high + 1.0) / 2.0
        win_cross_entropy = \
            0.5 * tf.nn.sigmoid_cross_entropy_with_logits(labels = winrate_dt_low,
                                                          logits = 2 * halflogits_nn_low) + \
            0.5 * tf.nn.sigmoid_cross_entropy_with_logits(labels = winrate_dt_high,
                                                          logits = 2 * halflogits_nn_high)
        # winrate = (outcome_dt + 1.0) / 2.0
        # win_cross_entropy = \
        #     tf.nn.sigmoid_cross_entropy_with_logits(labels = winrate,
        #                                             logits = 2 * alpha_nn)

        kle_loss = 2.0 * tf.reduce_mean(win_cross_entropy)
        if self.kle_loss_wt > 0.0:
            loss += self.kle_loss_wt * kle_loss

        # Loss on alpha*beta
        axb_diff = tf.multiply(beta_dt, tf.math.subtract(alpha_dt, alpha_nn))
        axb_loss = self.axb_scale \
                   * (tf.reduce_mean(tf.math.softplus(2 * axb_diff) - axb_diff) - tf.math.softplus(0.0))
        # axb_diff = ( 1.0 - outcome_nn * outcome_nn ) * halfbeta_nn * ( alpha_dt - alpha_nn )
        # axb_loss = self.axb_scale * tf.reduce_mean( axb_diff * axb_diff )

        if self.axb_loss_wt > 0.0:
            loss += self.axb_loss_wt * axb_loss

        # This 'policy layer calibration' term helps the tower to avoid coordinates shifts
        # 'w_fc_1' is the 722x362 matrix of weights of the dense layer in the policy head
        # By attracting it to plttarget matrix, we enforce the calibration of the 1x1 conv layer
        # This should help all the tower.
        if self.pol_layer_calib > 0.0:
            print("Policy layer calibration active with coefficient {}".format(self.pol_layer_calib))
            w_fc1 = [v for v in tf.trainable_variables() if v.name == "fp32_storage/w_fc_1:0"][0]
            pltarget = 0.5 * tf.concat([tf.eye(BOARD_SQUARES,1+BOARD_SQUARES),tf.eye(BOARD_SQUARES,1+BOARD_SQUARES)],axis=0)    
            plc_term = self.l2_scale * tf.nn.l2_loss(w_fc1 - pltarget)
            loss += self.reg_loss_wt * self.pol_layer_calib * plc_term
            reg_term += self.pol_layer_calib * plc_term   # this is done to see the term inside regularization
        
        return loss, policy_loss, mse_loss, kle_loss, axb_loss, reg_term, policylogits_nn

    def assign(self, var, values):
        try:
            self.session.run(tf.assign(var, values))
        except:
            print("Failed to assign {}: var shape {}, values shape {}".format(
                var.name, var.shape, values.shape))
            raise

    def replace_weights(self, new_weights):
        for e, weights in enumerate(self.weights):
            if isinstance(weights, str):
                weights = tf.get_default_graph().get_tensor_by_name(weights)
            if weights.name.endswith('/batch_normalization/beta:0'):
                # Batch norm beta is written as bias before the batch
                # normalization in the weight file for backwards
                # compatibility reasons.
                bias = tf.constant(new_weights[e], shape=weights.shape)
                # Weight file order: bias, means, variances
                var = tf.constant(new_weights[e + 2], shape=weights.shape)
                new_beta = tf.divide(bias, tf.sqrt(var + tf.constant(1e-5)))
                self.assign(weights, new_beta)
            elif weights.shape.ndims == 4:
                # Convolution weights need a transpose
                #
                # TF (kYXInputOutput)
                # [filter_height, filter_width, in_channels, out_channels]
                #
                # Leela/cuDNN/Caffe (kOutputInputYX)
                # [output, input, filter_size, filter_size]
                s = weights.shape.as_list()
                shape = [s[i] for i in [3, 2, 0, 1]]
                new_weight = tf.constant(new_weights[e], shape=shape)
                self.assign(weights, tf.transpose(new_weight, [2, 3, 1, 0]))
            elif weights.shape.ndims == 2:
                # Fully connected layers are [in, out] in TF
                #
                # [out, in] in Leela
                #
                s = weights.shape.as_list()
                shape = [s[i] for i in [1, 0]]
                new_weight = tf.constant(new_weights[e], shape=shape)
                self.assign(weights, tf.transpose(new_weight, [1, 0]))
            else:
                # Biases, batchnorm etc
                new_weight = tf.constant(new_weights[e], shape=weights.shape)
                self.assign(weights, new_weight)
        #This should result in identical file to the starting one
        #self.save_leelaz_weights('restored.txt')

    def restore(self, file):
        print("Restoring from {0}".format(file))
        optimistic_restore(self.session, file)

    def measure_loss(self, batch, training=False):
        # Measure loss over one batch. If training is true, also
        # accumulate the gradient and increment the global step.
        ops = [self.policy_loss, self.mse_loss, self.kle_loss, self.axb_loss, self.reg_term, self.accuracy ]
        if training:
            ops += [self.grad_op, self.step_op],
        r = self.session.run(ops, feed_dict={self.training: training,
                           self.planes: batch[0],
                           self.probs: batch[1],
                           self.komi: batch[2],
                           self.winner: batch[3],
                           self.alpha: batch[4],
                           self.beta: batch[5]})
        # Google's paper scales mse by 1/4 to a [0,1] range, so we do the same here
        return {'policy': r[0], 'mse': r[1]/4., 'kle': r[2], 'axb': r[3], 'reg': r[4],
                'accuracy': r[5], 'total': r[0]+r[1]+r[2]+r[3]+r[4] }

    def process(self, train_data, test_data):
        info_steps = INFO_STEPS
        stats = Stats()
        timer = Timer()
        n = 0
        while n < self.max_steps or self.max_steps == 0:
            n += 1
            batch = next(train_data)
            # Measure losses and compute gradients for this batch.
            losses = self.measure_loss(batch, training=True)
            stats.add(losses)
            # fetch the current global step.
            steps = tf.train.global_step(self.session, self.global_step)
            if n % self.macrobatch == (self.macrobatch-1):
                # Apply the accumulated gradients to the weights.
                self.session.run([self.train_op])
                # Clear the accumulated gradient.
                self.session.run([self.clear_op])

            if n % info_steps == 0:
                speed = info_steps * self.batch_size / timer.elapsed()
                print("step {}, policy={:g} mse={:g} kle={:g} axb={:g} reg={:g} total={:g} ({:g} pos/s)".format(
                    steps, stats.mean('policy'), stats.mean('mse'), stats.mean('kle'), stats.mean('axb'), stats.mean('reg'),
                    stats.mean('total'), speed))
                summaries = stats.summaries({'Policy Loss': 'policy',
                                             'MSE Loss': 'mse',
                                             'KLE Loss': 'kle',
                                             'Alpha*Beta Loss': 'axb'})
                self.train_writer.add_summary(
                    tf.Summary(value=summaries), steps)
                stats.clear()

                if n % (10 * info_steps) == 0:
                    print('Computing validation loss...')
                    val_stats = Stats()
                    val_batches = info_steps
                    for _ in range(0, val_batches):
                        val_batch = next(test_data)
                        losses = self.measure_loss(val_batch, training=False)
                        val_stats.add(losses)
                    summaries = val_stats.summaries({'Policy Loss': 'policy',
                                                     'MSE Loss': 'mse',
                                                     'KLE Loss': 'kle',
                                                     'Alpha*Beta Loss': 'axb',
                                                     'Accuracy': 'accuracy'})
                    self.test_writer.add_summary(tf.Summary(value=summaries), steps)
                    print("step {}, policy={:g} training accuracy={:g}%, mse={:g}, kle={:g}, axb={:g}".\
                          format(steps, val_stats.mean('policy'),
                                 val_stats.mean('accuracy')*100.0,
                                 val_stats.mean('mse'),
                                 val_stats.mean('kle'),
                                 val_stats.mean('axb')))
                    val_stats.clear()
                    if USE_TRAINSET_TEST:
                        for _ in range(0, val_batches):
                            val_batch = next(train_data)
                            losses = self.measure_loss(val_batch, training=False)
                            val_stats.add(losses)
                        summaries = val_stats.summaries({'Policy Loss': 'policy',
                                                         'MSE Loss': 'mse',
                                                         'KLE Loss': 'kle',
                                                         'Alpha*Beta Loss': 'axb',
                                                         'Accuracy': 'accuracy'})
                        self.trset_test_writer.add_summary(tf.Summary(value=summaries), steps)
                        print("step {}, policy={:g} training accuracy={:g}%, mse={:g}, kle={:g}, axb={:g}".\
                              format(steps, val_stats.mean('policy'),
                                     val_stats.mean('accuracy')*100.0,
                                     val_stats.mean('mse'),
                                     val_stats.mean('kle'),
                                     val_stats.mean('axb')))
                        val_stats.clear()

                    if (n >= self.min_steps and (n - self.min_steps) % self.train_steps == 0) \
                       or os.path.exists('request-save-model'):
                        print("Saving model...")
                        # Write out current model and checkpoint
                        path = os.path.join(os.getcwd(), "leelaz-model")
                        save_path = self.saver.save(self.session, path,
                                                    global_step=steps)
                        print("Model saved in file: {}".format(save_path))
                        print("Saving weights...")
                        leela_path = path + "-" + str(steps) + ".txt"
                        self.save_leelaz_weights(leela_path)
                        print("SAI weights saved to {}".format(leela_path))
                        # Things have likely changed enough
                        # that stats are no longer valid.

                        if self.swa_enabled:
                            self.save_swa_network(steps, path, leela_path, train_data)

                            save_path = self.saver.save(self.session, path,
                                                        global_step=steps)
                            print("Model saved in file: {}".format(save_path))

                        if os.path.exists('request-save-model'):
                            print("Stopping.")
                            os._exit(0)


        print("Finished.")
        os._exit(0)

    def save_leelaz_weights(self, filename):
        with open(filename, "w") as file:
            # Version tag
            file.write(WEIGHTS_FILE_VER)
            for weights in self.weights:
                work_weights = None
                if weights.name.endswith('/batch_normalization/gamma:0'):
                    continue
                elif weights.name.endswith('/batch_normalization/beta:0'):
                    # Batch norm beta needs to be converted to biases before
                    # the batch norm for backwards compatibility reasons
                    var_key = weights.name.replace('beta', 'moving_variance')
                    var = tf.get_default_graph().get_tensor_by_name(var_key)
                    gamma_key = weights.name.replace('beta', 'gamma')
                    try:
                        gamma = tf.get_default_graph().get_tensor_by_name(gamma_key)
                    except KeyError:
                        gamma = 1.0
                    work_weights = tf.divide(tf.multiply(weights, tf.sqrt(var + tf.constant(1e-5))),
                                             gamma)
                elif weights.name.endswith('/batch_normalization/moving_variance:0'):
                    gamma_key = weights.name.replace('moving_variance', 'gamma')
                    try:
                        gamma = tf.get_default_graph().get_tensor_by_name(gamma_key)
                    except KeyError:
                        gamma = 1.0
                    work_weights = tf.divide(weights, tf.square(gamma + tf.constant(1e-5)))
                elif weights.shape.ndims == 4:
                    # Convolution weights need a transpose
                    #
                    # TF (kYXInputOutput)
                    # [filter_height, filter_width, in_channels, out_channels]
                    #
                    # Leela/cuDNN/Caffe (kOutputInputYX)
                    # [output, input, filter_size, filter_size]
                    work_weights = tf.transpose(weights, [3, 2, 0, 1])
                elif weights.shape.ndims == 2:
                    # Fully connected layers are [in, out] in TF
                    #
                    # [out, in] in Leela
                    #
                    work_weights = tf.transpose(weights, [1, 0])
                else:
                    # Biases, batchnorm etc
                    work_weights = weights
                nparray = work_weights.eval(session=self.session)
                wt_str = [str(wt) for wt in np.ravel(nparray)]
                # Newline unless last line (single bias)
                file.write("\n")
                file.write(" ".join(wt_str))

    def save_gamma_weights(self, filename):
        with open(filename, "w") as file:
            for weights in self.weights:
                if not weights.name.endswith('/batch_normalization/gamma:0'):
                    continue
                nparray = weights.eval(session=self.session)
                wt_str = [str(wt) for wt in np.ravel(nparray)]
                # Newline unless last line (single bias)
                file.write(" ".join(wt_str))
                file.write("\n")

    def get_batchnorm_key(self):
        result = "bn" + str(self.batch_norm_count)
        self.batch_norm_count += 1
        return result

    def reset_batchnorm_key(self):
        self.batch_norm_count = 0
        self.reuse_var = True

    def add_weights(self, var):
        if self.reuse_var is None:
            if var.name[-11:] == "fp16_cast:0":
                name = var.name[:-12] + ":0"
                var = tf.get_default_graph().get_tensor_by_name(name)
            # All trainable variables should be stored as fp32
            assert var.dtype.base_dtype == tf.float32
            self.weights.append(var)

    def batch_norm(self, net, scale=False):
        # The weights are internal to the batchnorm layer, so apply
        # a unique scope that we can store, and use to look them back up
        # later on.
        scope = self.get_batchnorm_key()
        with tf.variable_scope(scope,
                               custom_getter=float32_variable_storage_getter):
            net = tf.layers.batch_normalization(
                    net,
                    epsilon=1e-5, axis=1, fused=True,
                    center=True, scale=scale,
                    training=self.training,
                    reuse=self.reuse_var)

        weight_vars = ['beta', 'moving_mean', 'moving_variance' ]
        if scale:
            weight_vars += ['gamma']
        for v in weight_vars:
            name = "fp32_storage/" + scope + '/batch_normalization/' + v + ':0'
            var = tf.get_default_graph().get_tensor_by_name(name)
            self.add_weights(var)

        return net

    def dense_block(self, inputs, input_channels, output_channels, name,
                    bias=True, relu=True, batch_norm=False):
        W_fc = weight_variable("w_" + name,
                               [input_channels, output_channels],
                               self.model_dtype)
        self.add_weights(W_fc)
        h_fc = tf.matmul(inputs, W_fc)

        if bias and not batch_norm:
            b_fc = bias_variable("b_" + name, [output_channels], self.model_dtype)
            self.add_weights(b_fc)
            h_fc = tf.add(h_fc, b_fc)

        if batch_norm:
            h_fc = self.batch_norm(h_fc)

        if relu:
            h_fc = tf.nn.relu(h_fc)

        return h_fc

    def resdense_block(self, inputs, channels, name):
        W_fc1 = weight_variable("w1_" + name, [channels, channels],
                                self.model_dtype)
        self.add_weights(W_fc1)
        h_fc = tf.matmul(inputs, W_fc1)
        h_fc = self.batch_norm(h_fc)
        h_fc = tf.nn.relu(h_fc)

        W_fc2 = weight_variable("w2_" + name, [channels, channels],
                                self.model_dtype)
        self.add_weights(W_fc2)
        h_fc = tf.matmul(h_fc, W_fc2)
        h_fc = self.batch_norm(h_fc, scale=RESIDUAL_BN_SCALE)
        h_fc = tf.add(h_fc, inputs)
        h_fc = tf.nn.relu(h_fc)

        return h_fc

    def conv_block(self, inputs, filter_size, input_channels, output_channels, name):
        W_conv = weight_variable(
            name,
            [filter_size, filter_size, input_channels, output_channels],
            self.model_dtype)

        self.add_weights(W_conv)

        net = inputs
        net = conv2d(net, W_conv)
        net = self.batch_norm(net)
        net = tf.nn.relu(net)
        return net

    def resconv_block(self, inputs, channels, name, filter_size=3):
        net = inputs
        orig = tf.identity(net)

        # First convnet weights
        W_conv_1 = weight_variable(name + "_conv_1", [filter_size, filter_size, channels, channels],
                                   self.model_dtype)
        self.add_weights(W_conv_1)

        net = conv2d(net, W_conv_1)
        net = self.batch_norm(net)
        net = tf.nn.relu(net)

        # Second convnet weights
        W_conv_2 = weight_variable(name + "_conv_2", [filter_size, filter_size, channels, channels],
                                   self.model_dtype)
        self.add_weights(W_conv_2)

        net = conv2d(net, W_conv_2)
        net = self.batch_norm(net, scale=RESIDUAL_BN_SCALE)
        net = tf.add(net, orig)
        net = tf.nn.relu(net)

        return net

    def construct_net(self, planes, komi):
        # NCHW format
        # batch, 18 channels, BOARD_SIZE x BOARD_SIZE
        x_planes = tf.reshape(planes, [-1, INPUT_PLANES + 1 + INPUT_STM, BOARD_SIZE, BOARD_SIZE])
        x_komi = tf.reshape(komi, [-1, 1])

        # Input convolution
        flow = self.conv_block(x_planes, filter_size=3,
                               input_channels=INPUT_PLANES + 1 + INPUT_STM,
                               output_channels=self.residual_filters,
                               name="first_conv")
        # Residual tower
        for i in range(0, self.resconv_blocks):
            block_name = "res_" + str(i)
            flow = self.resconv_block(flow, self.residual_filters,
                                       name=block_name)

        # POLICY

        policy_flow = flow

        # Policy blocks (optional)
        for i in range(0, POLICY_BLOCKS):
            block_inputs = POLICY_CHANNELS if i > 0 else self.residual_filters
            block_outputs = POLICY_CHANNELS
            if block_inputs != block_outputs:
                block_name = "policy_conv_" + str(i)
                print("Block {} inputs {} outputs {}".format(block_name, block_inputs, block_outputs))
                policy_flow = self.conv_block(policy_flow, filter_size=1,
                                              input_channels=block_inputs,
                                              output_channels=block_outputs,
                                              name=block_name)
            else:
                block_name = "policy_res_" + str(i)
                print("Block {} outputs {}".format(block_name, block_outputs))
                policy_flow = self.resconv_block(policy_flow, block_outputs, filter_size=1,
                                                  name=block_name)

        # Policy head
        block_inputs = POLICY_CHANNELS if POLICY_BLOCKS > 0 else self.residual_filters
        conv_pol = self.conv_block(policy_flow, filter_size=1,
                                   input_channels=block_inputs,
                                   output_channels=POLICY_OUTPUTS,
                                   name="policy_head")
        h_conv_pol_flat = tf.reshape(conv_pol, [-1, POLICY_OUTPUTS * BOARD_SQUARES])

        POLICY_DENSE_INPUTS = POLICY_OUTPUTS * BOARD_SQUARES

        W_fc1 = weight_variable("w_fc_1",[POLICY_DENSE_INPUTS, BOARD_SQUARES + 1], self.model_dtype)
        b_fc1 = bias_variable("b_fc_1",[BOARD_SQUARES + 1], self.model_dtype)
        self.add_weights(W_fc1)
        self.add_weights(b_fc1)
        h_fc1 = tf.add(tf.matmul(h_conv_pol_flat, W_fc1), b_fc1)

        # Value head - alpha
        conv_val = self.conv_block(flow, filter_size=1,
                                   input_channels=self.residual_filters,
                                   output_channels=VAL_OUTPUTS,
                                   name="value_conv")
        if not USE_ONLY_GLOBAL_STATS:
            value_initial_channels = VAL_OUTPUTS * BOARD_SQUARES
            value_flow = tf.reshape(conv_val, [-1, value_initial_channels])
        else:
            value_flow = self.conv_block(conv_val, filter_size=1,
                                       input_channels=VAL_OUTPUTS,
                                       output_channels=VALUE_GLOBAL_STATS,
                                       name="value_pooling_conv")
            value_flow = tf.reduce_mean(value_flow, axis=[2,3], name="value_pooling")
            value_initial_channels = VALUE_GLOBAL_STATS


        # Value blocks (optional)
        for i in range(0, VALUE_BLOCKS):
            if i==0 and value_initial_channels != VALUE_CHANNELS:
                block_name = "value_dense_" + str(i)
                value_flow = self.dense_block(value_flow,
                                              input_channels=value_initial_channels,
                                              output_channels=VALUE_CHANNELS,
                                              batch_norm=True,
                                              name=block_name)
            else:
                block_name = "value_resdense_" + str(i)
                value_flow = self.resdense_block(value_flow, channels=VALUE_CHANNELS,
                                                name=block_name)

        block_inputs = VALUE_CHANNELS if VALUE_BLOCKS > 0 else value_initial_channels
        h_fc2 = self.dense_block(value_flow,
                                 input_channels=block_inputs,
                                 output_channels=VAL_CHANS,
                                 name="fc_2")
        # W_fc2 = weight_variable("w_fc_2",[VAL_OUTPUTS * BOARD_SQUARES, VAL_CHANS], self.model_dtype)
        # b_fc2 = bias_variable("b_fc_2",[VAL_CHANS], self.model_dtype)
        # self.add_weights(W_fc2)
        # self.add_weights(b_fc2)
        # h_fc2 = tf.nn.relu(tf.add(tf.matmul(h_conv_val_flat, W_fc2), b_fc2))


        if VALUE_HEAD_TYPE == DOUBLE_I:
            value_head_rets = 2
        else:
            value_head_rets = 1

        h_fc3 = self.dense_block(h_fc2,
                                 input_channels=VAL_CHANS,
                                 output_channels=value_head_rets,
                                 relu=False,
                                 name="fc_3")
        # W_fc3 = weight_variable("w_fc_3",[VAL_CHANS, value_head_rets], self.model_dtype)
        # b_fc3 = bias_variable("b_fc_3",[value_head_rets], self.model_dtype)
        # self.add_weights(W_fc3)
        # self.add_weights(b_fc3)
        # h_fc3 = tf.add(tf.matmul(h_fc2, W_fc3), b_fc3)

        scale_factor = tf.constant(10.0 / BOARD_SQUARES / self.beta_scale)

        if VALUE_HEAD_TYPE == SINGLE:
            alpha = tf.add(h_fc3, -x_komi)
            halfbeta = 1.0

        # Value head - beta is 2 * halfbeta because one computes tanh(halfbeta * (alpha + kt))

        else:
            if VALUE_HEAD_TYPE == DOUBLE_I:
                h_fc3 = h_fc3[0]                                      # correct? wrong?
                h_fc5 = h_fc3[1]                                      # correct? wrong?

            elif VALUE_HEAD_TYPE == DOUBLE_T:
                W_fc5 = weight_variable("w_fc_5",[VAL_CHANS, VBE_NUM], self.model_dtype)
                b_fc5 = bias_variable("b_fc_5",[VBE_NUM], self.model_dtype)
                self.add_weights(W_fc5)
                self.add_weights(b_fc5)
                h_fc5 = tf.add(tf.matmul(h_fc2, W_fc5), b_fc5)

            elif VALUE_HEAD_TYPE == DOUBLE_Y:
                h_fc4 = self.dense_block(value_flow,
                                         input_channels=block_inputs,
                                         output_channels=VBE_CHANS,
                                         name="fc_4")
                # W_fc4 = weight_variable("w_fc_4",[VAL_OUTPUTS * BOARD_SQUARES, VBE_CHANS], self.model_dtype)
                # self.add_weights(W_fc4)
                # b_fc4 = bias_variable("b_fc_4",[VBE_CHANS], self.model_dtype)
                # self.add_weights(b_fc4)
                # h_fc4 = tf.nn.relu(tf.add(tf.matmul(value_flow, W_fc4), b_fc4))

                h_fc5 = self.dense_block(h_fc4,
                                         input_channels=VBE_CHANS,
                                         output_channels=VBE_NUM,
                                         relu=False,
                                         name="fc_5")
                # W_fc5 = weight_variable("w_fc_5",[VBE_CHANS, VBE_NUM], self.model_dtype)
                # b_fc5 = bias_variable("b_fc_5",[VBE_NUM], self.model_dtype)
                # self.add_weights(W_fc5)
                # self.add_weights(b_fc5)
                # h_fc5 = tf.add(tf.matmul(h_fc4, W_fc5), b_fc5)

            elif VALUE_HEAD_TYPE == DOUBLE_V:
                conv_vbe = self.conv_block(flow, filter_size=1,
                                           input_channels=self.residual_filters,
                                           output_channels=VBE_OUTPUTS,
                                           name="vbe_head")
                h_conv_vbe_flat = tf.reshape(conv_vbe, [-1, VBE_OUTPUTS * BOARD_SQUARES])

                W_fc4 = weight_variable("w_fc_4",[VBE_OUTPUTS * BOARD_SQUARES, VBE_CHANS], self.model_dtype)
                b_fc4 = bias_variable("b_fc_4",[VBE_CHANS], self.model_dtype)
                self.add_weights(W_fc4)
                self.add_weights(b_fc4)
                h_fc4 = tf.nn.relu(tf.add(tf.matmul(h_conv_vbe_flat, W_fc4), b_fc4))

                W_fc5 = weight_variable("w_fc_5",[VBE_CHANS, VBE_NUM], self.model_dtype)
                b_fc5 = bias_variable("b_fc_5",[VBE_NUM], self.model_dtype)
                self.add_weights(W_fc5)
                self.add_weights(b_fc5)
                h_fc5 = tf.add(tf.matmul(h_fc4, W_fc5), b_fc5)

            if QUARTILE_ENCODING:
                if VBE_NUM == 1:
                    # single sigmoid, the two heads are alpha25 and alpha75
                    alpha = 0.5 * ( h_fc3 + h_fc5 )
                    interq_range = h_fc5 - h_fc3
                else:
                    # piecewise sigmoid, alpha is alpha and the second head holds the quartiles
                    alpha = h_fc3
                    interq_range = (h_fc5 - h_fc3) * np.array([-1,1], ndmin=2)

                eps = 0.05
                log3 = 1.0986122886681096913952452369225
                halfbeta = log3 / ( eps + tf.nn.relu(interq_range) )
            else:
                alpha = h_fc3
                halfbeta = tf.scalar_mul(scale_factor, tf.exp(h_fc5))



        return h_fc1, alpha, halfbeta         # policy-logits, alpha, beta/2

    def snap_save(self):
        # Save a snapshot of all the variables in the current graph.
        if not hasattr(self, 'save_op'):
            save_ops = []
            rest_ops = []
            for var in self.weights:
                if isinstance(var, str):
                    var = tf.get_default_graph().get_tensor_by_name(var)
                name = var.name.split(':')[0]
                v = tf.Variable(var, name='save/'+name, trainable=False)
                save_ops.append(tf.assign(v, var))
                rest_ops.append(tf.assign(var, v))
            self.save_op = tf.group(*save_ops)
            self.restore_op = tf.group(*rest_ops)
        self.session.run(self.save_op)

    def snap_restore(self):
        # Restore variables in the current graph from the snapshot.
        self.session.run(self.restore_op)

    def save_swa_network(self, steps, path, leela_path, data):
        # Sample 1 in self.swa_c of the networks. Compute in this way so
        # that it's safe to change the value of self.swa_c
        rem = self.session.run(tf.assign_add(self.swa_skip, -1))
        if rem > 0:
            return
        self.swa_skip.load(self.swa_c, self.session)

        # Add the current weight vars to the running average.
        num = self.session.run(self.swa_accum_op)

        if self.swa_max_n != None:
            num = min(num, self.swa_max_n)
            self.swa_count.load(float(num), self.session)

        swa_path = path + "-swa-" + str(int(num)) + "-" + str(steps) + ".txt"

        # save the current network.
        self.snap_save()
        # Copy the swa weights into the current network.
        self.session.run(self.swa_load_op)
        if self.swa_recalc_bn:
            print("Refining SWA batch normalization")
            for _ in range(200):
                batch = next(data)
                self.session.run(
                    [self.loss, self.update_ops],
                    feed_dict={self.training: True,
                               self.planes: batch[0], self.probs: batch[1],
                               self.winner: batch[2]})

        self.save_leelaz_weights(swa_path)
        # restore the saved network.
        self.snap_restore()

        print("Wrote averaged network to {}".format(swa_path))

# Unit tests for TFProcess.
def gen_block(size, f_in, f_out):
    return [ [1.1] * size * size * f_in * f_out, # conv
             [-.1] * f_out,  # bias weights
             [-.2] * f_out,  # batch norm mean
             [-.3] * f_out ] # batch norm var

class TFProcessTest(unittest.TestCase):
    def test_can_replace_weights(self):
        tfprocess = TFProcess(6, 128)
        tfprocess.init(batch_size=1)
        # use known data to test replace_weights() works.
        data = gen_block(3, 18, tfprocess.residual_filters) # input conv
        for _ in range(tfprocess.resconv_blocks):
            data.extend(gen_block(3,
                tfprocess.residual_filters, tfprocess.residual_filters))
            data.extend(gen_block(3,
                tfprocess.residual_filters, tfprocess.residual_filters))
        # policy
        data.extend(gen_block(1, tfprocess.residual_filters, 2))
        data.append([0.4] * 2*BOARD_SQUARES * (BOARD_SQUARES+1))
        data.append([0.5] * (BOARD_SQUARES+1))
        # value
        data.extend(gen_block(1, tfprocess.residual_filters, 1))
        data.append([0.6] * BOARD_SQUARES * 256)
        data.append([0.7] * 256)
        data.append([0.8] * 256)
        data.append([0.9] * 1)
        tfprocess.replace_weights(data)

if __name__ == '__main__':
    unittest.main()
