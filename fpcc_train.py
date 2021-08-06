import argparse
import tensorflow as tf
import numpy as np
import os
import time
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print (BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, '../../'))
sys.path.append(os.path.join(BASE_DIR, '../../utils'))
sys.path.append(os.path.join(BASE_DIR, '../../models'))
import provider
from utils.test_utils import *
from models import model

# Parsing Arguments
parser = argparse.ArgumentParser()
# Experiment Settings
parser.add_argument('--gpu', type=str, default="0", help='GPU to use [default: GPU 0]')
parser.add_argument('--wd', type=float, default=0.9, help='Weight Decay [Default: 0.0]')
parser.add_argument('--epoch', type=int, default=100, help='Number of epochs [default: 50]')
parser.add_argument('--batch', type=int, default=4, help='Batch Size during training [default: 4]')
parser.add_argument('--point_num', type=int, default=4096, help='Point Number')
parser.add_argument('--group_num', type=int, default=50, help='Maximum Group Number in one pc')
# parser.add_argument('--cate_num', type=int, default=1, help='Number of categories')
parser.add_argument('--margin_same', type=float, default=0.5, help='loss margin: same instance')
parser.add_argument('--margin_diff', type=float, default=1., help='loss margin: different instance')
parser.add_argument('--use_vdm', type=bool, default=True, help='use the valid distance matrix for loss')
parser.add_argument('--use_asm', type=bool, default=True, help='use the attention score matrix for loss')

parser.add_argument('--backbone', type=str, default='dgcnn', help='backbone: pointnet,dgcnn')

# Input&Output Settings
parser.add_argument('--point_dim', type=int, default=6, help='dim of point cloud,[XYZ],[XYZ,NxNyNz], [XYZ,NxNyNz,RGB]')
parser.add_argument('--output_dir', type=str, default='checkpoint/', help='Directory that stores all training logs and trained models')
parser.add_argument('--input_list', type=str, default='datas/ring_train.txt', help='Input data list file')
parser.add_argument('--d_max', type=float, default=0.18, help=' ring: 0.18 / gear: 0.25')
parser.add_argument('--restore_model', type=str, default='checkpoint/', help='Pretrained model')

FLAGS = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

TRAINING_FILE_LIST = FLAGS.input_list
PRETRAINED_MODEL_PATH = os.path.join(FLAGS.restore_model, 'ring_vdm_asm/')

POINT_DIM = FLAGS.point_dim
POINT_NUM = FLAGS.point_num
BATCH_SIZE = FLAGS.batch
OUTPUT_DIR = FLAGS.output_dir
D_MAX = FLAGS.d_max
vdm = FLAGS.use_vdm
asm = FLAGS.use_asm
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

NUM_GROUPS = FLAGS.group_num
# NUM_CATEGORY = FLAGS.cate_num

print('#### Batch Size: {0}'.format(BATCH_SIZE))
print('#### Point Number: {0}'.format(POINT_NUM))
print('#### Training using GPU: {0}'.format(FLAGS.gpu))

DECAY_STEP = 800000.
DECAY_RATE = 0.5

LEARNING_RATE_CLIP = 1e-6
BASE_LEARNING_RATE = 1e-4
MOMENTUM = 0.9
BACKBONE = FLAGS.backbone


TRAINING_EPOCHES = FLAGS.epoch
# MARGINS = [FLAGS.margin_same, FLAGS.margin_diff]
MARGINS = [FLAGS.margin_same, FLAGS.margin_diff]

print('### Training epoch: {0}'.format(TRAINING_EPOCHES))

# MODEL_STORAGE_PATH = os.path.join(OUTPUT_DIR, 'hv6_scoreweight')
MODEL_STORAGE_PATH = PRETRAINED_MODEL_PATH
if not os.path.exists(MODEL_STORAGE_PATH):
    os.mkdir(MODEL_STORAGE_PATH)

LOG_STORAGE_PATH = os.path.join(OUTPUT_DIR, 'logs')
if not os.path.exists(LOG_STORAGE_PATH):
    os.mkdir(LOG_STORAGE_PATH)

SUMMARIES_FOLDER = os.path.join(OUTPUT_DIR, 'summaries')
if not os.path.exists(SUMMARIES_FOLDER):
    os.mkdir(SUMMARIES_FOLDER)

LOG_DIR = FLAGS.output_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)

def printout(flog, data):
    print(data)
    flog.write(data + '\n')

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(FLAGS.gpu)):
            batch = tf.Variable(0, trainable=False, name='batch')
            learning_rate = tf.train.exponential_decay(
                BASE_LEARNING_RATE,  # base learning rate
                batch * BATCH_SIZE,  # global_var indicating the number of steps
                DECAY_STEP,  # step size
                DECAY_RATE,  # decay rate
                staircase=True  # Stair-case or continuous decreasing
            )
            learning_rate = tf.maximum(learning_rate, LEARNING_RATE_CLIP)

            lr_op = tf.summary.scalar('learning_rate', learning_rate)

            pointclouds_ph, ptsgroup_label_ph, pts_score_ph = \
                model.placeholder_inputs(BATCH_SIZE, POINT_NUM, NUM_GROUPS, POINT_DIM)
            is_training_ph = tf.placeholder(tf.bool, shape=())

            labels = {'ptsgroup': ptsgroup_label_ph,
                      # 'semseg': ptsseglabel_ph,
                      # 'semseg_mask': pts_seglabel_mask_ph,
                      # 'group_mask': pts_group_mask_ph,
                      'center_score': pts_score_ph}

            net_output = model.get_model(BACKBONE, pointclouds_ph, is_training_ph)
            loss, score_loss, grouperr = model.get_loss(net_output, labels,vdm, asm, D_MAX, MARGINS)

            total_training_loss_ph = tf.placeholder(tf.float32, shape=())
            group_err_loss_ph = tf.placeholder(tf.float32, shape=())
            total_train_loss_sum_op = tf.summary.scalar('total_training_loss', total_training_loss_ph)
            group_err_op = tf.summary.scalar('group_err_loss', group_err_loss_ph)

        train_variables = tf.trainable_variables()

        trainer = tf.train.AdamOptimizer(learning_rate)
        train_op = trainer.minimize(loss, var_list=train_variables, global_step=batch)

        loader = tf.train.Saver([v for v in tf.all_variables()#])
                                 if
                                   ('conf_logits' not in v.name) and
                                    ('Fsim' not in v.name) and
                                    ('Fsconf' not in v.name) and
                                    ('batch' not in v.name)
                                ])

        saver = tf.train.Saver([v for v in tf.all_variables()])

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        init = tf.global_variables_initializer()
        sess.run(init)

        train_writer = tf.summary.FileWriter(SUMMARIES_FOLDER + '/train', sess.graph)

        train_file_list = provider.getDataFiles(TRAINING_FILE_LIST)
        num_train_file = len(train_file_list)

        fcmd = open(os.path.join(LOG_STORAGE_PATH, 'cmd.txt'), 'w')
        fcmd.write(str(FLAGS))
        fcmd.close()

        log_file = time.strftime('log_%Y-%m-%d_%H-%M-%S', time.gmtime())
        flog = open(os.path.join(LOG_STORAGE_PATH, log_file + '.txt'), 'w')

        ckptstate = tf.train.get_checkpoint_state(PRETRAINED_MODEL_PATH)
        if ckptstate is not None:
            LOAD_MODEL_FILE = os.path.join(PRETRAINED_MODEL_PATH, os.path.basename(ckptstate.model_checkpoint_path))
            loader.restore(sess, LOAD_MODEL_FILE)
            printout(flog, "Model loaded in file: %s" % LOAD_MODEL_FILE)
        else:
            printout(flog, "Fail to load modelfile: %s" % PRETRAINED_MODEL_PATH)


        train_file_idx = np.arange(0, len(train_file_list))
        np.random.shuffle(train_file_idx)

        ## load all data into memory
        all_data = []
        all_group = []
        all_seg = []
        all_score = []
        for i in range(num_train_file):
            cur_train_filename = train_file_list[train_file_idx[i]]
            printout(flog, 'Loading train file ' + cur_train_filename +'\t'+ str(i)+'/'+str(num_train_file))
            cur_data, cur_group, _, _, cur_score = provider.loadDataFile_with_groupseglabel_stanfordindoor(cur_train_filename)
            # cur_data = cur_data.reshape([-1,4096,3])

            all_data += [cur_data]
            all_group += [cur_group]
            all_score += [cur_score]

        all_data = np.concatenate(all_data,axis=0)
        all_group = np.concatenate(all_group,axis=0)
        all_score = np.concatenate(all_score,axis=0)


        num_data = all_data.shape[0]
        num_batch = num_data // BATCH_SIZE

        def train_one_epoch(epoch_num):

            ### NOTE: is_training = False: 
            ### do not update bn parameters during training due to the small batch size. This requires pre-training PointNet with large batchsize (say 32).
            is_training = False

            order = np.arange(num_data)
            np.random.shuffle(order)

            total_loss = 0.0
            total_score_loss = 0.0
            total_grouperr = 0.0


            for j in range(num_batch):
                begidx = j * BATCH_SIZE
                endidx = (j + 1) * BATCH_SIZE

                # pts_label_one_hot, pts_label_mask = model.convert_seg_to_one_hot(all_seg[order[begidx: endidx]])
                pts_group_label, _ = model.convert_groupandcate_to_one_hot(all_group[order[begidx: endidx]],NUM_GROUPS=NUM_GROUPS)                
                pts_score = all_score[order[begidx: endidx]]
                input_data = all_data[order[begidx: endidx], ...]

                feed_dict = {
                    pointclouds_ph: input_data[...,:POINT_DIM],
                    ptsgroup_label_ph: pts_group_label,
                    pts_score_ph:pts_score,
                    is_training_ph: is_training,
                }

                _, loss_val,score_loss_val, grouperr_val = sess.run([train_op, loss, score_loss, grouperr], feed_dict=feed_dict)

                total_loss += loss_val
                total_score_loss += score_loss_val


                total_grouperr += grouperr_val


                if j % 100 == 99:
                    printout(flog, 'Batch: %d, loss: %f, score_loss: %f, grouperr: %f' % (j, total_loss/100, total_score_loss/100, total_grouperr/100))

                    lr_sum, batch_sum, train_loss_sum, group_err_sum = sess.run( \
                        [lr_op, batch, total_train_loss_sum_op, group_err_op], \
                        feed_dict={total_training_loss_ph: total_loss / 100.,
                                   group_err_loss_ph: total_grouperr / 100., })

                    train_writer.add_summary(train_loss_sum, batch_sum)
                    train_writer.add_summary(lr_sum, batch_sum)
                    train_writer.add_summary(group_err_sum, batch_sum)

                    total_grouperr = 0.0
                    total_loss = 0.0
                    total_score_loss = 0.0


        if not os.path.exists(MODEL_STORAGE_PATH):
            os.mkdir(MODEL_STORAGE_PATH)

        for epoch in range(TRAINING_EPOCHES):
            printout(flog, '\n>>> Training for the epoch %d/%d ...' % (epoch, TRAINING_EPOCHES))

            train_file_idx = np.arange(0, len(train_file_list))
            np.random.shuffle(train_file_idx)

            train_one_epoch(epoch)
            flog.flush()

            cp_filename = saver.save(sess, os.path.join(MODEL_STORAGE_PATH, 'epoch_' + str(epoch + 1) + '.ckpt'))
            printout(flog, 'Successfully store the checkpoint model into ' + cp_filename)


        flog.close()


if __name__ == '__main__':
    train()
