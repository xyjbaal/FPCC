import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../models'))
import tensorflow as tf
import numpy as np
import tf_util
import pointnet
import dgcnn



# NUM_CATEGORY = 1
# NUM_GROUPS = 100

def placeholder_inputs(batch_size, num_point, num_group, dim):

    if num_point == 0:
        pointclouds_ph = tf.placeholder(tf.float32, shape=(batch_size, None, dim))
    else:
        pointclouds_ph = tf.placeholder(tf.float32, shape=(batch_size, num_point, dim))

    pts_grouplabels_ph = tf.placeholder(tf.float32, shape=(batch_size, num_point, num_group))
    # pts_group_mask_ph = tf.placeholder(tf.float32, shape=(batch_size, num_point))

    # alpha_ph = tf.placeholder(tf.float32, shape=())
    pts_score_ph = tf.placeholder(tf.float32, shape=(batch_size, num_point))

    # return pointclouds_ph, pts_grouplabels_ph, pts_seglabel_mask_ph, pts_group_mask_ph, alpha_ph, pts_score_ph
    return pointclouds_ph, pts_grouplabels_ph, pts_score_ph
def convert_seg_to_one_hot(labels):
    # labels: semantic label BxN
    # as if B is batch size and N is the numbers of point 4096
    '''
    Returns
        label_one_hot: shap (batch_size,the number of point, num_category)
        pts_label_mask shap (B,N)
        
    '''

    label_one_hot = np.zeros((labels.shape[0], labels.shape[1], NUM_CATEGORY))
    pts_label_mask = np.zeros((labels.shape[0], labels.shape[1]))

    # return the unique item in array and its times, arrange from small to large
    # cnt is number of times each unique item appears in araary
    un, cnt = np.unique(labels, return_counts=True)

    label_count_dictionary = dict(zip(un, cnt))
    totalnum = 0

    # as if iteritems had be replaced with items in Python 3.X
    # for k_un, v_cnt in label_count_dictionary.iteritems():
    for k_un, v_cnt in label_count_dictionary.items():
        if k_un != -1:
            totalnum += v_cnt

    for idx in range(labels.shape[0]):
        for jdx in range(labels.shape[1]):
            if labels[idx, jdx] != -1:
                label_one_hot[idx, jdx, labels[idx, jdx]] = 1
                pts_label_mask[idx, jdx] = float(totalnum) / float(label_count_dictionary[labels[idx, jdx]]) 
                # pts_label_mask[idx, jdx] = 1. - float(label_count_dictionary[labels[idx, jdx]]) / totalnum

    return label_one_hot, pts_label_mask


def convert_groupandcate_to_one_hot(grouplabels, NUM_GROUPS=50):
    # grouplabels: BxN instance lable or pid

    group_one_hot = np.zeros((grouplabels.shape[0], grouplabels.shape[1], NUM_GROUPS))
    pts_group_mask = np.zeros((grouplabels.shape[0], grouplabels.shape[1]))

    un, cnt = np.unique(grouplabels, return_counts=True)
    group_count_dictionary = dict(zip(un, cnt))
    totalnum = 0

    # as if iteritems had be replaced with items in Python 3.X
    # for k_un, v_cnt in group_count_dictionary.iteritems():
    for k_un, v_cnt in group_count_dictionary.items():
        if k_un != -1:
            totalnum += v_cnt

    for idx in range(grouplabels.shape[0]):
        un = np.unique(grouplabels[idx])
        grouplabel_dictionary = dict(zip(un, range(len(un))))
        for jdx in range(grouplabels.shape[1]):
            if grouplabels[idx, jdx] != -1:
                group_one_hot[idx, jdx, grouplabel_dictionary[grouplabels[idx, jdx]]] = 1
                # pts_group_mask[idx, jdx] = float(totalnum) / float(group_count_dictionary[grouplabels[idx, jdx]]) 
                pts_group_mask[idx, jdx] = 1. - float(group_count_dictionary[grouplabels[idx, jdx]]) / totalnum

    return group_one_hot.astype(np.float32), pts_group_mask



def generate_group_mask(pts, grouplabels, labels):
    # grouplabels: BxN
    # pts: BxNx6
    # labels: BxN

    group_mask = np.zeros((grouplabels.shape[0], grouplabels.shape[1], grouplabels.shape[1]))

    for idx in range(grouplabels.shape[0]):
        for jdx in range(grouplabels.shape[1]):
            for kdx in range(grouplabels.shape[1]):
                if (labels[idx, jdx] == labels[idx, kdx]):
                    group_mask[idx, jdx, kdx] = 2.

                if np.linalg.norm((pts[idx, jdx, :3] - pts[idx, kdx, :3]) * (
                    pts[idx, jdx, :3] - pts[idx, kdx, :3])) < 0.04:
                    if (labels[idx, jdx] == labels[idx, kdx]):
                        group_mask[idx, jdx, kdx] = 5.
                    else:
                        group_mask[idx, jdx, kdx] = 2.

    return group_mask



def get_model(backbone, point_cloud, is_training, bn_decay=None, train=True):
    #input: point_cloud: BxNx9 (XYZ, RGB, NormalizedXYZ)

    # batch_size = point_cloud.get_shape()[0].value
    print(point_cloud.get_shape())
    # print('backbone is ',backbone)
    p_distance = None

    if backbone == 'dgcnn':
        F , p_distance = dgcnn.get_model(point_cloud, is_training, bn_decay=bn_decay)
    elif backbone == 'pointnet':
        F = pointnet.get_model(point_cloud, is_training, bn_decay=bn_decay)
    else:
        print('the backbone:'+backbone+ 'is error')

    # center prediction
    Center = tf_util.conv2d(F, 128, [1, 1], padding='VALID', stride=[1, 1], bn=False, is_training=is_training, scope='Center')

    ptscenter_logits = tf_util.conv2d(Center, 1, [1, 1], padding='VALID', stride=[1, 1], activation_fn=None, scope='conf_logits')
    # ptscenter_logits = tf.squeeze(ptscenter_logits, [2])
    ptscenter_logits = tf.squeeze(ptscenter_logits)
    ptscenter = tf.nn.sigmoid(ptscenter_logits,name="center_confidence")


    # ptssemseg = tf.nn.softmax(ptssemseg_logits, name="ptssemseg")

    # Similarity matrix
    Fsim = tf_util.conv2d(F, 128, [1, 1], padding='VALID', stride=[1, 1], bn=False, is_training=is_training, scope='Fsim')

    Fsim = tf.squeeze(Fsim, [2])

    if train == True:
        batch_size = point_cloud.get_shape()[0].value
        r = tf.reduce_sum(Fsim * Fsim, 2)
        r = tf.reshape(r, [batch_size, -1, 1])
        D = r - 2 * tf.matmul(Fsim, tf.transpose(Fsim, perm=[0, 2, 1])) + tf.transpose(r, perm=[0, 2, 1])

        # simmat_logits = tf.maximum(m*D, 0.)
        # simmat is the Similarity Matrix
        simmat_logits = tf.maximum(D, 0.)
    else:
        simmat_logits = None

    # Confidence Map
    # Fconf = tf_util.conv2d(F, 128, [1, 1], padding='VALID', stride=[1, 1], bn=False, is_training=is_training, scope='Fsconf')
    # conf_logits = tf_util.conv2d(Fconf, 1, [1, 1], padding='VALID', stride=[1, 1], activation_fn=None, scope='conf_logits')
    # conf_logits = tf.squeeze(conf_logits, [2])

    # conf = tf.nn.sigmoid(conf_logits, name="confidence")

    return {'center_score': ptscenter,
            'point_features':Fsim,
            'simmat': simmat_logits,
            '3d_distance':p_distance}

def get_loss(net_output, labels, vdm=True, asm=True, d_max=1, margin=[.5, 1.]):
    """
    input:
        net_output:{'center_score', 'point_features','simmat','3d_distance'}
        labels:{'ptsgroup', 'center_score', 'group_mask'}
    """
    dis_matrix = net_output['3d_distance'] # (B,N,N)
    pred_simmat = net_output['simmat']

    pts_group_label = labels['ptsgroup']
    pts_score_label = labels['center_score']


    # Similarity Matrix loss
    B = pts_group_label.get_shape()[0]
    N = pts_group_label.get_shape()[1]

    onediag = tf.ones([B,N], tf.float32)


    group_mat_label = tf.matmul(pts_group_label,tf.transpose(pts_group_label, perm=[0, 2, 1])) #BxNxN: (i,j) = 1   if i and j in the same group
    group_mat_label = tf.matrix_set_diag(group_mat_label,onediag)

    valid_distance_matrix = tf.cast(tf.less(dis_matrix, tf.constant(d_max)), tf.float32)


    # sem_mat_label = tf.cast(tf.matmul(pts_semseg_label,tf.transpose(pts_semseg_label, perm=[0, 2, 1])), tf.float32) #BxNxN: (i,j) = 1 if i and j are the same semantic category
    # sem_mat_label = tf.matrix_set_diag(sem_mat_label,onediag)
    # sem_mat_label = tf.ones([B,N,N], tf.float32)

    # samesem_mat_label = sem_mat_label
    # diffsem_mat_label = tf.subtract(1.0, sem_mat_label)# 1- same semantic matirx --> different sematix matrix

    samegroup_mat_label = group_mat_label   # same instance matrix
    diffgroup_mat_label = tf.subtract(1.0, group_mat_label) # diferent instance matrix
    
    if vdm == True:
        diffgroup_mat_label = tf.multiply(diffgroup_mat_label, valid_distance_matrix)# diferent instance matrix * valid distance


    # diffgroup_samesem_mat_label = tf.multiply(diffgroup_mat_label, samesem_mat_label) # different instance but same semantic
    # diffgroup_diffsem_mat_label = tf.multiply(diffgroup_mat_label, diffsem_mat_label) # different instance and same semantic
    diffgroup_samesem_mat_label = diffgroup_mat_label # FPCC does not deel with different objetcs


    num_samegroup = tf.reduce_sum(samegroup_mat_label)
    # num_diffgroup_samesem = tf.reduce_sum(diffgroup_samesem_mat_label)
    # num_diffgroup_diffsem = tf.reduce_sum(diffgroup_diffsem_mat_label)

    # Double hinge loss

    # C_same = tf.constant(margin[0], name="C_same") # same semantic category
    # C_diff = tf.constant(margin[1], name="C_diff") # different semantic category

    # pos =  tf.multiply(samegroup_mat_label, pred_simmat) # minimize distances if in the same group
    # neg_samesem = tf.multiply(diffgroup_samesem_mat_label, tf.maximum(tf.subtract(C_same, pred_simmat), 0))
    # neg_diffsem = tf.multiply(diffgroup_diffsem_mat_label, tf.maximum(tf.subtract(C_diff, pred_simmat), 0))

    C_same = tf.constant(margin[0], name="C_same") # same semantic category
    C_diff = tf.constant(margin[1], name="C_diff") # different semantic category
    # C_same_ins = tf.constant(margin[2], name="C_same_ins")

    # pos =  tf.multiply(samegroup_mat_label, pred_simmat) # minimize distances if in the same group
    same_ins_loss         = 2*tf.multiply(samegroup_mat_label, tf.maximum(tf.subtract(pred_simmat,C_same), 0))  # minimize distances if in the same instance
    diff_ins_loss = tf.multiply(diffgroup_samesem_mat_label, tf.maximum(tf.subtract(C_diff, pred_simmat), 0)) # maximum distances if in the diff instance


    # simmat_loss = alpha * neg_samesem + pos
    simmat_loss =  same_ins_loss + diff_ins_loss
    # group_mask_weight = tf.matmul(group_mask, tf.transpose(group_mask, perm=[0, 2, 1]))

    # attention score matrix
    score_mask = pts_score_label
    score_mask = tf.expand_dims(score_mask, dim=-1)
    multiples = tf.constant([1,1,4096], tf.int32) 
    score_mask = tf.tile(score_mask, multiples)
    score_mask = tf.add(score_mask,tf.transpose(score_mask, perm=[0, 2, 1]))
    score_mask = tf.clip_by_value(score_mask, clip_value_min=0, clip_value_max=1)

    if asm == True:
        simmat_loss = tf.multiply(simmat_loss, score_mask)

    simmat_loss = tf.reduce_mean(simmat_loss)

    # loss of center point

    sigma_squared = 2
    regression_diff = tf.subtract(net_output['center_score'],pts_score_label)
    regression_diff = tf.abs(regression_diff)
    regression_loss = tf.where(
            tf.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * tf.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

    ptscenter_loss = tf.reduce_mean(regression_loss)

    ng_label = group_mat_label
    ng_label = tf.greater(ng_label, tf.constant(0.5)) 

    ng = tf.less(pred_simmat, tf.constant(margin[0]))


    loss = simmat_loss + 3*ptscenter_loss

    grouperr = tf.abs(tf.cast(ng, tf.float32) - tf.cast(ng_label, tf.float32))
    # 计算分group错误的点的数量。group_error

    return loss, ptscenter_loss,tf.reduce_mean(grouperr) 