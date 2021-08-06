import os
import numpy as np
import glob
import csv
import h5py

root_dir  = './IPARingScrew_part_1/train_pointcloud/*.txt'
matrix_dir = './IPARingScrew_part_1/gt/'
output_dir = os.path.join('./', 'IPARingScrew_part_1/h5')
data_dtype = 'float32'
label_dtype = 'int32'
def fpcc_save_h5(h5_filename, data,  data_dtype='float32', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename)

    p_xyz = data[...,:6]
    gid = data[:,:,-2]
    center_score = data[:,:,-1]

    h5_fout.create_dataset(
            'data', data=p_xyz,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'center_score', data=center_score,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'gid', data=gid,
            compression='gzip', compression_opts=4,
            dtype=label_dtype)
    h5_fout.close()

def samples(data, sample_num_point, dim=None):

    if dim == None:
        dim = data.shape[-1]
    N = data.shape[0]
    order = np.arange(N)
    np.random.shuffle(order)
    data = data[order, :]


    batch_num = int(np.ceil(N / float(sample_num_point)))
    sample_datas = np.zeros((batch_num, sample_num_point, dim))


    for i in range(batch_num):
        beg_idx = i*sample_num_point
        end_idx = min((i+1)*sample_num_point, N)
        num = end_idx - beg_idx
        sample_datas[i,0:num,:] = data[beg_idx:end_idx, :]

        if num < sample_num_point:
            makeup_indices = np.random.choice(N, sample_num_point - num)
            sample_datas[i,num:,:] = data[makeup_indices, :]
    return sample_datas

def samples_plus_normalized(data_label, num_point):

    data = data_label
    dim = data.shape[-1]
    # print('dim',dim)

    xyz_min = np.amin(data_label, axis=0)[0:3]
    data[:, 0:3] -= xyz_min

    max_x = max(data[:,0])
    max_y = max(data[:,1])
    max_z = max(data[:,2])

    data_batch = samples(data, num_point, dim=dim)
    # print('label_batch',label_batch,np.max(label_batch))
    new_data_batch = np.zeros((data_batch.shape[0], num_point, dim+3))
    for b in range(data_batch.shape[0]):
        new_data_batch[b, :, 3] = data_batch[b, :, 0]/max_x
        new_data_batch[b, :, 4] = data_batch[b, :, 1]/max_y
        new_data_batch[b, :, 5] = data_batch[b, :, 2]/max_z

    new_data_batch[:, :, 0:3] = data_batch[:,:,0:3]
    new_data_batch[:, :, 6:] = data_batch[:,:,3:]

    return new_data_batch

# gear: 0.08 ring: 0.1 
# A: 0.5 B: 0.8 C: 0.4
r_max = 0.1
for f in glob.glob(root_dir):
    file_name = f.split("\\")[-1].split('.')[0]
    cycle = file_name[:4]
    index = file_name[-3:]
    if int(index)<11:
        continue
    gt_dir = matrix_dir+'cycle_'+cycle+'/'+index+'.csv'
    # gt = np.loadtxt(gt_dir)
    gt_data = []
    with open(gt_dir) as csvfile:
        csv_reader = csv.reader(csvfile)  
        birth_header = next(csv_reader)  
        for row in csv_reader:
            gt_data.append(row)
        gt_data = [[float(x) for x in row[2:5]] for row in gt_data]  


    scene_txt = np.loadtxt(f)
    # scene_txt is [n x 4]: x,y,z, inslab

    num = int(index)
    parts_index = scene_txt[:,-1]
    score_all = []
    part_all = []
    for i in range(1,num+1):
        part_ = scene_txt[np.where(parts_index == i)]
        if len(part_) == 0:
            continue
        part_all.append(part_)
        c = gt_data[i]
        distance = np.sqrt(np.sum((part_[:,:3]-c)**2,axis=1))
        score = 1-(distance/r_max)**2
        score = np.clip(score,0,1)
        score_all.append(score)
    score_all = np.concatenate(score_all)
    part_all = np.concatenate(part_all)

    part_all = part_all.reshape(-1,4)
    score_all = score_all.reshape(-1,1)

    new_scene_txt =  np.concatenate((part_all, score_all),axis=1)
    new_scene_txt[:,:3]

    # np.savetxt('./IPAGearShaft/train_pointcloud/point_%s_%s.txt' % (cycle,index),new_scene_txt)

    h5_filename = os.path.join(output_dir, '%s_%s.h5' % (cycle,index))
    if os.path.exists(h5_filename):
        continue
    data = samples_plus_normalized(new_scene_txt, 4096)

    print('data',data.shape)
    fpcc_save_h5(h5_filename, data, data_dtype, label_dtype)


