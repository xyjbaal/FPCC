import argparse
import tensorflow as tf
import json
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
import sys
from scipy import stats

# import matplotlib.pyplot as plt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, '../../'))
sys.path.append(os.path.join(BASE_DIR, '../../utils'))
sys.path.append(os.path.join(BASE_DIR, '../../models'))
import provider
from utils.test_utils import *
from models import model
import glob

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', type=str, default="0", help='GPU to use [default: GPU 1]')
parser.add_argument('--verbose', action='store_true', help='if specified, use depthconv')
# parser.add_argument('--input_list', type=str, default='t', help='test data list')
parser.add_argument('--restore_dir', type=str, default='checkpoint/', help='Directory that stores all training logs and trained models')
parser.add_argument('--point_dim', type=int, default=6, help='dim of point cloud,XYZ,NxNyNz or RGB')
parser.add_argument('--conf_th', type=float, default=0.6, help='min valid confidence 0.4~0.8')
parser.add_argument('--backbone', type=str, default='dgcnn', help='backbone: pointnet,dgcnn,Xnet')
parser.add_argument('--r_nms', type=float, default=.1, help='bunny:, A:.6 / B: 1 / C:1.2 / ring: 0.1 / gear: .08')
FLAGS = parser.parse_args()

# DEFAULT SETTINGS
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
PRETRAINED_MODEL_PATH = os.path.join(FLAGS.restore_dir,'ring_vdm_asm/')
OUTPUT_DIR = os.path.join('./test_results/ring_vdm_asm/')

TEST_DIR = './datas/ring_test/*.txt'
# OUTPUT_VERBOSE = FLAGS.verbose  # If true, output all color-coded segmentation obj files
OUTPUT_VERBOSE = True


R_NMS = FLAGS.r_nms
BACKBONE = FLAGS.backbone
SAMPLE_LIMIT = None
# In fact, for some tasks, e.g., pose estimation, XYZ location, you do not need to process all points.


conf_threshold = FLAGS.conf_th
max_feature_distance = None # The point whose feature distance from the center point is greater than this value is regarded as noise [-1]
max_3d_distance = 1. # The farthest distance from the point to the center. usually max_3d_distance > r_nms


RESTORE_DIR = FLAGS.restore_dir
gpu_to_use = FLAGS.gpu
POINT_DIM = FLAGS.point_dim
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

OUTPUT_DIR_2 = os.path.join(OUTPUT_DIR, 'scene_seg')
if not os.path.exists(OUTPUT_DIR_2):
    os.mkdir(OUTPUT_DIR_2)

OUTPUT_DIR_3 = os.path.join(OUTPUT_DIR, 'center_map/')
if not os.path.exists(OUTPUT_DIR_3):
    os.mkdir(OUTPUT_DIR_3)



# MAIN SCRIPT

POINT_NUM = 4096
# POINT_NUM = 8192
if SAMPLE_LIMIT is None:
    BATCH_SIZE  = 20
else:
    BATCH_SIZE  = SAMPLE_LIMIT




def printout(flog, data):
    print(data)
    flog.write(data + '\n')

def samples(data, sample_num_point,limit=None):

    N = data.shape[0]
    dim =  data.shape[-1]
    order = np.arange(N)
    np.random.shuffle(order)

    data = data[order, :]

    if limit == None:
        batch_num = int(np.ceil(N / float(sample_num_point)))
    else:
        batch_num = min(int(np.ceil(N / float(sample_num_point))),limit)

    sample_datas = np.zeros((batch_num, sample_num_point, dim))


    for i in range(batch_num):
        beg_idx = i*sample_num_point
        end_idx = min((i+1)*sample_num_point, N)
        num = end_idx - beg_idx
        sample_datas[i,0:num,:] = data[beg_idx:end_idx, :]

        if num < sample_num_point:
        	# print('makeup')
        	makeup_indices = np.random.choice(N, sample_num_point - num)
        	sample_datas[i,num:,:] = data[makeup_indices, :]

    return sample_datas

def samples_reshape_txt(data_label, num_point=4096,limit=None):
    """ input: [X,Y,Z]  shape：（N,3）or [X,Y,Z, inslab] (N,4)
        for XYZ, add normalized XYZ as 678 channels and aligned XYZ as 345 channels

        return:
        x,y,z,x0,y0,z0, Nx,Ny,Nz
    """
    dim = data_label.shape[-1]
    xyz_min = np.amin(data_label, axis=0)[0:3]
    xyz_max = np.amax(data_label, axis=0)[0:3]



    data_align = np.zeros((data_label.shape[0], dim+6))

    xyz_min = np.amin(data_label, axis=0)[0:3]
    data_align[:,0:3] = data_label[:,0:3]
    data_align[:,3:6] = data_label[:,0:3]-xyz_min
    data_align[:,-1] = data_label[:,-1]
    data_align[:,-2] = data_label[:,-2]

    max_x = max(data_align[:,3])
    max_y = max(data_align[:,4])
    max_z = max(data_align[:,5])

    data_batch  = samples(data_align, num_point,limit)
    batch_num = data_batch.shape[0]


    new_data_batch = np.zeros((batch_num, num_point, 9))

    for b in range(batch_num):
        new_data_batch[b, :, 6] = data_batch[b, :, 3]/max_x
        new_data_batch[b, :, 7] = data_batch[b, :, 4]/max_y
        new_data_batch[b, :, 8] = data_batch[b, :, 5]/max_z


    new_data_batch[:, :, 0:6] = data_batch[:,:,0:6]
    gt =  data_batch[:,:,-1]

    return new_data_batch, gt


def predict():
    is_training = False

    with tf.device('/gpu:' + str(gpu_to_use)):
        is_training_ph = tf.placeholder(tf.bool, shape=())

        pointclouds_ph, _, _ = \
            model.placeholder_inputs(BATCH_SIZE, POINT_NUM, 50, POINT_DIM)

        net_output = model.get_model(BACKBONE, pointclouds_ph,is_training_ph,train=False)

       

    # Add ops to save and restore all the variables.

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:

        flog = open(os.path.join(OUTPUT_DIR, 'log.txt'), 'w')

        # Restore variables from disk.

        ckptstate = tf.train.get_checkpoint_state(PRETRAINED_MODEL_PATH)
        if ckptstate is not None:
            LOAD_MODEL_FILE = os.path.join(PRETRAINED_MODEL_PATH,os.path.basename(ckptstate.model_checkpoint_path))
            saver.restore(sess, LOAD_MODEL_FILE)
            printout(flog, "Model loaded in file: %s" % LOAD_MODEL_FILE)
        else:
            printout(flog, "Fail to load modelfile: %s" % PRETRAINED_MODEL_PATH)




        un_gt_list = []
        output_filelist_f = os.path.join(OUTPUT_DIR, 'output_filelist.txt')
        fout_out_filelist = open(output_filelist_f, 'w')
        for f in glob.glob(TEST_DIR):
            file_name = f.split("\\")[-1].split('.')[0]
            print(file_name)
            if not os.path.exists(f):
                print('%s is not exists',f)
                continue
            # scene_start = time.time()
            points = np.loadtxt(f)[:,:] # points: [XYZ,instance label]
            points_num = points.shape[0]

            input_data, gt_batch = samples_reshape_txt(points,num_point=POINT_NUM,limit=SAMPLE_LIMIT)
            # input_data： [original XYZ, aligned XYZ, normalizated XYZ] N x 9 (without RGB)
              

            out_data_label_filename = file_name + '_pred.txt'
            out_data_label_filename = os.path.join(OUTPUT_DIR, out_data_label_filename)
            out_gt_label_filename = file_name + '_gt.txt'
            out_gt_label_filename = os.path.join(OUTPUT_DIR, out_gt_label_filename)
            fout_data_label = open(out_data_label_filename, 'w')
            fout_gt_label = open(out_gt_label_filename, 'w')
            fout_out_filelist.write(out_data_label_filename+'\n')



            valid_batch_num = input_data.shape[0]
            predict_num = int(np.ceil(valid_batch_num / BATCH_SIZE))
            point_features = []
            pts_scores = []

            for n in range(predict_num):
                feed_data = np.zeros((BATCH_SIZE, POINT_NUM, POINT_DIM))
                beg_idx = n * BATCH_SIZE
                end_idx = min((n+1)*BATCH_SIZE, valid_batch_num)
                num = end_idx - beg_idx
                feed_data[:num,:,:] = input_data[beg_idx:end_idx,:,3:3+POINT_DIM]


                feed_dict = {
                pointclouds_ph: feed_data,
                is_training_ph: is_training,
                }


                point_feature, pts_score_val0= \
                sess.run([net_output['point_features'],
                net_output['center_score']],
                feed_dict=feed_dict)

                point_features.append([point_feature])
                pts_scores.append([pts_score_val0])

            pred_score_val = np.concatenate(pts_scores,axis=0)
            point_features = np.concatenate(point_features,axis=0)



            
            input_data = input_data.reshape([-1, 3+POINT_DIM])
            input_data = input_data[:points_num,:]

            pred_score_val = pred_score_val.reshape([-1,1])
            pred_score_val = pred_score_val[:points_num]

            point_features = point_features.reshape([-1,128])
            point_features = point_features[:points_num,:]

            group_pred, c_index = GroupMerging_fpcc(input_data[:,3:6],point_features, pred_score_val, \
                conf_threshold=conf_threshold, max_feature_dis=max_feature_distance, use_3d_mask=max_3d_distance, r_nms=R_NMS)
            # scene_end = scene_start- time.time()

            # c_score = pred_score_val[c_index]

            pts = input_data


            ###### Generate Results for Evaluation
            group_pred_final = group_pred.reshape(-1)
            group_gt = gt_batch.reshape(-1)
            # seg_pred = np.zeros((group_pred.shape))

            ins_pre = group_pred_final.astype(np.int32)

            ins_gt = group_gt
            # un_gt = np.unique(ins_gt)
            # un_gt_list.append(len(un_gt))
            for i in range(pts.shape[0]):
                fout_data_label.write('%f %f %f %d\n' % (pts[i, 0], pts[i, 1], pts[i, 2], ins_pre[i]))
                fout_gt_label.write('%d\n' % ins_gt[i])

            fout_data_label.close()
            fout_gt_label.close()
            
            
            if OUTPUT_VERBOSE:
                output_color_point_cloud(pts[:, 3:6], ins_pre.astype(np.int32),
                	os.path.join(OUTPUT_DIR_2, '%s_grouppred.txt' % (file_name)))

                output_color_point_center_score(pts[:, 3:6], pred_score_val,os.path.join(OUTPUT_DIR_3,'%s_c_map.txt' % (file_name)))
 
        fout_out_filelist.close()

with tf.Graph().as_default():
    predict()
