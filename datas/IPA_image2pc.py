import numpy as np
import cv2
import json
import os
import h5py
from multiprocessing import Process


def perspectiveDepthImageToPointCloud(image_depth,image_rgb, seg, gt_path,defaultValue,perspectiveAngle,clip_start,clip_end,resolutionX,resolutionY,resolution_big,pixelOffset_X_KoSyTopLeft,pixelOffset_Y_KoSyTopLeft):
    '''
        Input: Depth image in perspective projection
        Output: Point cloud as list (in meter)
		
		Parameter:
        - image_depth: Depth image in perspective projection with shape (resolutionY,resolutionX,1)
        - defaultValue: Default value to indicate missing depth information in the depth image
        - perspectiveAngle: Perspective angle in deg
		- clip_start: Near clipping plane in meter
		- clip_end: Far clipping plane in meter
        - resolutionX: resolutionX of the input image
        - resolutionY: resolutionY of the input image
		- resolution_big: resolution_big of the input image
        - pixelOffset_X_KoSyTopLeft: Offset in x direction in pixel from coordinate system top left
        - pixelOffset_Y_KoSyTopLeft: Offset in y direction in pixel from coordinate system top left
    '''
    print('resolutionY,resolutionX',resolutionY,resolutionX)
    print('image_depth',image_depth.shape)
    assert(image_depth.shape==(resolutionY,resolutionX,1))
    # print('resolutionY,resolutionX',resolutionY,resolutionX)
    # Warning: Point cloud will not be correct when depth image was resized!
    
    image_big=np.zeros((resolution_big,resolution_big))
    image_big[pixelOffset_Y_KoSyTopLeft:pixelOffset_Y_KoSyTopLeft+resolutionY,pixelOffset_X_KoSyTopLeft:pixelOffset_X_KoSyTopLeft+resolutionX]=image_depth[:,:,0]
    image_depth=image_big
    image_depth=np.rot90(image_depth,k=2,axes=(0,1))
    # image_rgb = np.rot90(image_rgb, k=2, axes=(0,1))
    seg_big=np.zeros((resolution_big,resolution_big))
    seg_big[pixelOffset_Y_KoSyTopLeft:pixelOffset_Y_KoSyTopLeft+resolutionY,pixelOffset_X_KoSyTopLeft:pixelOffset_X_KoSyTopLeft+resolutionX]= seg[:,:,0]
    seg=seg_big
    seg = np.rot90(seg, k=2, axes=(0,1))
    print(image_rgb.shape)
    
    point_cloud=[]
    transforms = []
    range_=clip_end-clip_start
    print('gt_path',gt_path)
    with open(gt_path,'r',encoding='utf8')as fp:
    # json_data = json.load(fp)
        gt_info = json.load(fp)
    
	# Loop over all pixels in the depth image:
    # print('image_depth.shape',image_depth.shape)
    for j in range(image_depth.shape[0]):
        for i in range(image_depth.shape[1]):
            if image_depth[j,i]==defaultValue or image_depth[j,i]==0:
                # print('no depth')
                continue
            # r = image_rgb[j,i,0]
            # g = image_rgb[j,i,1]
            # b = image_rgb[j,i,2]
            label = seg[j,i]
            seg_id = int(label)-1
            if label == 255 or label == 0:
                continue
            world_z=(image_depth[j,i]*range_+clip_start)
            # Calculate the orthogonal size based on current depth (function of z value)
            orthoSizeZ_x=np.tan(np.deg2rad(perspectiveAngle/2))*world_z*2*resolutionX/resolution_big
            orthoSizeZ_y=np.tan(np.deg2rad(perspectiveAngle/2))*world_z*2*resolutionY/resolution_big
            
            meterPerPixel_x=orthoSizeZ_x/resolutionX
            meterPerPixel_y=orthoSizeZ_y/resolutionY
            
            world_x=(i+0.5-resolution_big/2)*meterPerPixel_x
            world_y=(j+0.5-resolution_big/2)*meterPerPixel_y
            
            # print('seg_id',seg_id)
            t = gt_info[seg_id]['t']
            t = np.array(t).reshape(-1)
            rotation = gt_info[seg_id]['R']
            rotation = np.array(rotation).reshape(-1)
            # print(rotation)
            # rotation = rotation.reshape(-1)
            visib = 1.0 - gt_info[seg_id]['occlusion_rate']
            # p=[world_x,world_y,world_z,r,g,b,rotation,t,visib,label]
            # r = [ _r,for _r in rotation]
            # t = [_t, for _t in t]
            trans = np.concatenate((rotation,t))
            # p=[world_x,world_y,world_z,r,g,b,visib,label]
            p=[world_x,world_y,world_z,label]
            transforms.append(trans)
            point_cloud.append(p)
    return point_cloud, transforms


part_dir = './SileaneBunny_part_1/'
with open(part_dir+'parameter.json', 'r') as f:
    parameter = json.load(f)
print(parameter)
cycle_ = parameter['number_cycles']
drop_ = parameter['shapeDropLimit']

defaultValue = -1
perspectiveAngle = parameter['perspectiveAngle']
clip_start = parameter['clip_start']
clip_end = parameter['clip_end']
resolutionX = parameter['resolutionX']
resolutionY = parameter['resolutionY']
resolution_big = parameter['resolution_big']
pixelOffset_X_KoSyTopLeft = parameter['pixelOffset_X_KoSyTopLeft']
pixelOffset_Y_KoSyTopLeft = parameter['pixelOffset_Y_KoSyTopLeft']


NUM_POINT = 4096
data_dtype = 'float32'
label_dtype = 'int32'
def process_data(data_range):
    for c in range(data_range[0], data_range[1]):
        for d in range(10,drop_+1):
            # if os.path.exists('./SileaneBunny/pointcloud/%04d_%03d.txt'%(c,d)):
            #     continue
            depth = cv2.imread(part_dir+"p_depth/cycle_%04d/%03d_depth_uint16.png"%(c,d))
            # print('depth',depth.shape)
            depth = depth[:,:,0:1]


            depth=np.array(depth,dtype='float')
            depth /= 255

            RGB = cv2.imread(part_dir+"p_rgb/cycle_%04d/%03d_rgb.png"%(c,d),-1)
            seg = cv2.imread(part_dir+"p_segmentation/cycle_%04d/%03d_segmentation.png"%(c,d),-1)
            print('depth',seg,seg.shape)
            # seg = seg.reshape([512,512,-1])
            seg = seg.reshape([474,506,-1])
            # seg = seg.reshape([1018,1178,-1])


            # defaultValue = 0
            # perspectiveAngle = 0
            # clip_start = 0
            # clip_end = 0
            gt_path = part_dir+"gt/cycle_%04d/%03d.json"%(c,d)
            # print('gt_path',gt_path)

            point_cloud , transforms = perspectiveDepthImageToPointCloud(depth,RGB,seg, gt_path, defaultValue,perspectiveAngle,clip_start,clip_end,resolutionX,resolutionY,resolution_big,pixelOffset_X_KoSyTopLeft,pixelOffset_Y_KoSyTopLeft)
            # print(point_cloud)
            np.savetxt('./SileaneBunny_part_1/train_pointcloud/%04d_%03d.txt'%(c,d), point_cloud, fmt='%0.6f')




TRAINING_DATA_NUM = 10
MAX_PROCESS = 1
if __name__ == '__main__':
    # process_data([0,250])

    data_per_process = int(TRAINING_DATA_NUM/MAX_PROCESS)
    total_data_range = []
    for i in range(MAX_PROCESS - 1):
        data_range = [i*data_per_process, (i+1)*data_per_process]
        total_data_range.append(data_range)
    total_data_range.append([(MAX_PROCESS - 1)*data_per_process, TRAINING_DATA_NUM])

    procs = []
    for index, data_range in enumerate(total_data_range):
        proc = Process(target = process_data, args = (data_range,))
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()