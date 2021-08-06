import h5py
import numpy as np



if __name__ == '__main__':
    f = h5py.File('./ring_train/0000_015.h5','r')
    print(list(f.keys()))
    data = f['data'][:]
    score=f['score'][:]
    # score=f['center_score'][:]
    label = f['pid'][:]
    # label = f['gid'][:]
    # print(label,label.shape)

    out_file = './ring_train_0000_015.txt'
    with open(out_file, 'w') as f:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                # f.write('%f %f %f %f %f %f %f %d\n' % (data[i][j][0], data[i][j][1], data[i][j][2],data[i][j][3], data[i][j][4], data[i][j][5],score[i][j], label[i][j]))
                f.write('%f %f %f %f %d\n' % (data[i][j][0], data[i][j][1], data[i][j][2], score[i][j], label[i][j]))

    
