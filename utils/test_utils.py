import numpy as np
from scipy import stats
import matplotlib
# import matplotlib as matplotlib

import json
import math
import time
matplotlib.use('Agg')
import matplotlib.cm

############################
##    Ths Statistics      ##
############################

def Get_Ths(pts_corr, seg, ins, ths, ths_, cnt):

    pts_in_ins = {}
    for ip, pt in enumerate(pts_corr):
        if ins[ip] in pts_in_ins.keys():
            pts_in_curins_ind = pts_in_ins[ins[ip]]
            pts_notin_curins_ind = (~(pts_in_ins[ins[ip]])) & (seg==seg[ip])
            hist, bin = np.histogram(pt[pts_in_curins_ind], bins=20)

            if seg[ip]==8:
                print (bin)

            numpt_in_curins = np.sum(pts_in_curins_ind)
            numpt_notin_curins = np.sum(pts_notin_curins_ind)

            if numpt_notin_curins > 0:

                tp_over_fp = 0
                ib_opt = -2
                for ib, b in enumerate(bin):
                    if b == 0:
                        break
                    tp = float(np.sum(pt[pts_in_curins_ind] < bin[ib])) / float(numpt_in_curins)
                    fp = float(np.sum(pt[pts_notin_curins_ind] < bin[ib])) / float(numpt_notin_curins)

                    if tp <= 0.5:
                        continue

                    if fp == 0. and tp > 0.5:
                        ib_opt = ib
                        break

                    if tp/fp > tp_over_fp:
                        tp_over_fp = tp / fp
                        ib_opt = ib

                if tp_over_fp >  4.:
                    ths[seg[ip]] += bin[ib_opt]
                    ths_[seg[ip]] += bin[ib_opt]
                    cnt[seg[ip]] += 1

        else:
            pts_in_curins_ind = (ins == ins[ip])
            pts_in_ins[ins[ip]] = pts_in_curins_ind
            pts_notin_curins_ind = (~(pts_in_ins[ins[ip]])) & (seg==seg[ip])
            hist, bin = np.histogram(pt[pts_in_curins_ind], bins=20)

            if seg[ip]==8:
                print (bin)

            numpt_in_curins = np.sum(pts_in_curins_ind)
            numpt_notin_curins = np.sum(pts_notin_curins_ind)

            if numpt_notin_curins > 0:

                tp_over_fp = 0
                ib_opt = -2
                for ib, b in enumerate(bin):

                    if b == 0:
                        break

                    tp = float(np.sum(pt[pts_in_curins_ind]<bin[ib])) / float(numpt_in_curins)
                    fp = float(np.sum(pt[pts_notin_curins_ind]<bin[ib])) / float(numpt_notin_curins)

                    if tp <= 0.5:
                        continue

                    if fp == 0. and tp > 0.5:
                        ib_opt = ib
                        break

                    if tp / fp > tp_over_fp:
                        tp_over_fp = tp / fp
                        ib_opt = ib

                if tp_over_fp >  4.:
                    ths[seg[ip]] += bin[ib_opt]
                    ths_[seg[ip]] += bin[ib_opt]
                    cnt[seg[ip]] += 1

    return ths, ths_, cnt
def distance_matrix(vector1,vector2):
    '''
    vector1 : (N1,d)
    vector2 : (N2,d)
    return : (N1,N2)
    '''
    r_1 = np.sum(np.square(vector1), axis=-1)
    r_2 = np.sum(np.square(vector2), axis=-1)
    r_2 = r_2.reshape([1,-1])
    r_1 = np.reshape(r_1, [-1, 1])
    s = 2 * np.matmul(vector1, np.transpose(vector2, axes=[1, 0]))
    dis_martix = r_1 - s + r_2

    return dis_martix

##############################
##    Merging Algorithms    ##
##############################
def GroupMerging_sgpn(pts_corr, confidence, seg, label_bin):

    confvalidpts = (confidence>0.4)
    un_seg = np.unique(seg)
    # refineseg = -1* np.ones(pts_corr.shape[0])
    groupid = -1* np.ones(pts_corr.shape[0])
    numgroups = 0
    groupseg = {}
    for i_seg in un_seg:
        if i_seg==-1:
            continue
        pts_in_seg = (seg==i_seg)
        valid_seg_group = np.where(pts_in_seg & confvalidpts)
        proposals = []
        if valid_seg_group[0].shape[0]==0:
            proposals += [pts_in_seg]
        else:
            for ip in valid_seg_group[0]:
                validpt = (pts_corr[ip] < label_bin[i_seg]) & pts_in_seg
                validpt = (pts_corr[ip] < 10) & pts_in_seg
                if np.sum(validpt)>5:
                    flag = False
                    for gp in range(len(proposals)):
                        iou = float(np.sum(validpt & proposals[gp])) / np.sum(validpt|proposals[gp])#uniou
                        validpt_in_gp = float(np.sum(validpt & proposals[gp])) / np.sum(validpt)#uniou
                        if iou > 0.5 or validpt_in_gp > 0.6:
                            flag = True
                            if np.sum(validpt)>np.sum(proposals[gp]):
                                proposals[gp] = validpt
                            continue

                    if not flag:
                        proposals += [validpt]

            if len(proposals) == 0:
                proposals += [pts_in_seg]
        for gp in range(len(proposals)):
            if np.sum(proposals[gp])>50:
                groupid[proposals[gp]] = numgroups
                groupseg[numgroups] = i_seg
                numgroups += 1
                # refineseg[proposals[gp]] = stats.mode(seg[proposals[gp]])[0]

    un, cnt = np.unique(groupid, return_counts=True)
    for ig, g in enumerate(un):
        if cnt[ig] < 50:
            groupid[groupid==g] = -1

    un, cnt = np.unique(groupid, return_counts=True)
    groupidnew = groupid.copy()
    for ig, g in enumerate(un):
        if g == -1:
            continue
        groupidnew[groupid==g] = (ig-1)
        groupseg[(ig-1)] = groupseg.pop(g)
    groupid = groupidnew

    for ip, gid in enumerate(groupid):
        if gid == -1:
            pts_in_gp_ind = (pts_corr[ip] < label_bin[seg[ip]])
            # pts_in_gp_ind = (pts_corr[ip] < 10)
            pts_in_gp = groupid[pts_in_gp_ind]
            pts_in_gp_valid = pts_in_gp[pts_in_gp!=-1]
            if len(pts_in_gp_valid) != 0:
                groupid[ip] = stats.mode(pts_in_gp_valid)[0][0]

    return groupid, 1, groupseg
def BlockMerging(volume, volume_seg, pts, grouplabel, groupseg, gap=1e-3):

    overlapgroupcounts = np.zeros([100,300])
    groupcounts = np.ones(100)
    x=(pts[:,0]/gap).astype(np.int32)
    y=(pts[:,1]/gap).astype(np.int32)
    z=(pts[:,2]/gap).astype(np.int32)
    for i in range(pts.shape[0]):
        xx=x[i]
        yy=y[i]
        zz=z[i]
        if grouplabel[i] != -1:
            if volume[xx,yy,zz]!=-1 and volume_seg[xx,yy,zz]==groupseg[grouplabel[i]]:
                overlapgroupcounts[grouplabel[i],volume[xx,yy,zz]] += 1
        groupcounts[grouplabel[i]] += 1

    groupcate = np.argmax(overlapgroupcounts,axis=1)
    maxoverlapgroupcounts = np.max(overlapgroupcounts,axis=1)

    curr_max = np.max(volume)
    for i in range(groupcate.shape[0]):
        if maxoverlapgroupcounts[i]<7 and groupcounts[i]>30:
            curr_max += 1
            groupcate[i] = curr_max


    finalgrouplabel = -1 * np.ones(pts.shape[0])

    for i in range(pts.shape[0]):
        if grouplabel[i] != -1 and volume[x[i],y[i],z[i]]==-1:
            volume[x[i],y[i],z[i]] = groupcate[grouplabel[i]]
            volume_seg[x[i],y[i],z[i]] = groupseg[grouplabel[i]]
            finalgrouplabel[i] = groupcate[grouplabel[i]]
    return finalgrouplabel

def GroupMerging_fpcc(pts, pts_features, center_scores, center_socre_th=0.5, max_feature_dis=None, use_3d_mask=None, r_nms=1):
    """
    input:
        pts: xyz of point cloud
        pts_features: 128-dim feature of each point Nx128
        center_scores： center_score of each pint Nx1 
    Returns:


    """

    validpts_index = np.where(center_scores > center_socre_th)[0]


    validpts = pts[validpts_index,:]
    validscore = center_scores[validpts_index]

    validscore = validscore.reshape(-1,1)
    validpts_index = validpts_index.reshape(-1,1)


    candidate_point_selected = np.concatenate((validpts,validscore, validpts_index),axis=1)

    heightest_point_selected = []

    validscore = validscore.reshape(-1)
    order = validscore.argsort()[::-1]
    center_points = []
    while order.size > 0:
        i = order[0]
        center_points.append(candidate_point_selected[i])
        distance = np.sqrt(np.sum((candidate_point_selected[order,:3]-candidate_point_selected[i,:3])**2,axis=1))
        remain_index = np.where(distance > r_nms)

        order = order[remain_index]

    center_points = np.concatenate(center_points, axis=0)
    center_points = center_points.reshape((-1,5))
    # print('number of instances',center_points.shape[0])

    center_index = np.array(center_points[:,-1]).astype(int)

    center_point_features = pts_features[center_index]

    pts_corr = distance_matrix(center_point_features,pts_features)

    if use_3d_mask is not None:
        pts_c = pts[center_index]

        dis_mask = distance_matrix(pts_c, pts)
        dis_mask = np.sqrt(dis_mask)
        pts_corr[np.where(dis_mask > use_3d_mask)] = 999

    groupid = np.argmin(pts_corr,axis=0)

    if max_feature_dis is not None:
        pts_corr_min = np.min(pts_corr, axis=0)
        over_threshold = np.where(pts_corr_min>max_feature_dis)
        groupid[over_threshold] = -1

    return groupid, center_index


############################
##    Evaluation Metrics  ##
############################

def eval_3d_perclass(tp, fp, npos):

    tp = np.asarray(tp).astype(np.float)
    fp = np.asarray(fp).astype(np.float)
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    rec = tp / npos
    prec = tp / (fp+tp)

    ap = 0.
    for t in np.arange(0, 1, 0.1):
        prec1 = prec[rec>=t]
        prec1 = prec1[~np.isnan(prec1)]
        if len(prec1) == 0:
            p = 0.
        else:
            p = max(prec1)
            if not p:
                p = 0.

        ap = ap + p / 10


    return ap, rec, prec

############################
##    Visualize Results   ##
############################

color_map = json.load(open('part_color_mapping.json', 'r'))
for i in range(len(color_map)):
    for k in range(len(color_map[i])):
        color_map[i][k] = math.floor(color_map[i][k]*255)
# print(color_map)

def output_bounding_box_withcorners(box_corners, seg, out_file):
    # ##############   0       1       2       3       4       5       6       7
    corner_indexes = [[0, 1, 2], [0, 1, 5], [0, 4, 2], [0, 4, 5], [3, 1, 2], [3, 1, 5], [3, 4, 2], [3, 4, 5]]
    line_indexes = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
    with open(out_file, 'w') as f:
        l = box_corners.shape[0]
        for i in range(l):
            box = box_corners[i]
            color = color_map[seg[i]]
            for line_index in line_indexes:
                corner0 = box[line_index[0]]
                corner1 = box[line_index[1]]
                print (corner0.shape)
                dist = np.linalg.norm(corner0 - corner1)
                dot_num = int(dist / 0.005)
                delta = (corner1 - corner0) / dot_num
                for idot in range(dot_num):
                    plotdot = corner0 + idot * delta
                    f.write(
                        'v %f %f %f %f %f %f\n' % (plotdot[0], plotdot[1], plotdot[2], color[0], color[1], color[2]))


def output_bounding_box(boxes, seg, out_file):
    # ##############   0       1       2       3       4       5       6       7
    #box:nx8x3
    corner_indexes = [[0, 1, 2], [0, 1, 5], [0, 4, 2], [0, 4, 5], [3, 1, 2], [3, 1, 5], [3, 4, 2], [3, 4, 5]]
    line_indexes = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
    with open(out_file, 'w') as f:
        l = boxes.shape[0]
        for i in range(l):
            box = boxes[i]
            color = color_map[seg[i]]
            for line_index in line_indexes:
                corner0 = box[corner_indexes[line_index[0]]]
                corner1 = box[corner_indexes[line_index[1]]]
                dist = np.linalg.norm(corner0 - corner1)
                dot_num = int(dist / 0.005)
                delta = (corner1 - corner0) / dot_num
                for idot in range(dot_num):
                    plotdot = corner0 + idot * delta
                    f.write(
                        'v %f %f %f %f %f %f\n' % (plotdot[0], plotdot[1], plotdot[2], color[0], color[1], color[2]))


def output_color_point_cloud(data, seg, out_file):
    with open(out_file, 'w') as f:
        l = len(seg)
        for i in range(l):
            color = color_map[seg[i]]
            # f.write('%f %f %f %d %d %d\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))
            # color[0] = math.floor(color[0]*255)
            # color[1] = math.floor(color[1]*255)
            # color[2] = math.floor(color[2]*255)
            # f.write('%f %f %f %d\n' % (data[i][0], data[i][1], data[i][2],i))
            f.write('%f %f %f %d %d %d\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))

def output_color_point_semantic(data, seg, out_file):
    l = len(seg)
    scene_num = 0
    un = np.unique(seg)
    out_file_name = out_file
    count = []
    # un = np.delete(un,0,axis=0)

    for j in range(len(un)):
        # if un[argdex[j]] == -1:
        #     continue
        index = np.where(seg==un[j])
        group = data[index]
        out_file = out_file_name+str(j)+'_'+str(scene_num)+'.txt'
        color = color_map[j]
        color = np.tile(color,(len(group),1))
        group_color = np.hstack((group,color))
        np.savetxt(out_file, group_color, fmt='%0.6f')

def output_color_point_center_score(data, c_s, out_file):

    c_s = c_s.reshape((-1,1))
    point_c_s = np.hstack((data,c_s))

    np.savetxt(out_file, point_c_s, fmt='%0.6f')


def output_point_cloud_rgb(data, rgb, out_file):
    with open(out_file, 'w') as f:
        l = len(data)
        for i in range(l):
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], rgb[i][0],  rgb[i][1],  rgb[i][2]))


def output_color_point_cloud_red_blue(data, seg, out_file):
    with open(out_file, 'w') as f:
        l = len(seg)
        for i in range(l):
            if seg[i] == 1:
                color = [0, 0, 1]
            elif seg[i] == 0:
                color = [1, 0, 0]
            else:
                color = [0, 0, 0]
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))


##define color heat map
norm = matplotlib.colors.Normalize(vmin=0, vmax=255)
magma_cmap = matplotlib.cm.get_cmap('magma')
magma_rgb = []
for i in range(0, 255):
       k = matplotlib.colors.colorConverter.to_rgb(magma_cmap(norm(i)))
       magma_rgb.append(k)


def output_scale_point_cloud(data, scales, out_file):
    with open(out_file, 'w') as f:
        l = len(scales)
        for i in range(l):
            scale = int(scales[i]*254)
            if scale > 254:
                scale = 254
            color = magma_rgb[scale]
            # f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))
            f.write('v %f %f %f %d %d %d\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))
