"""
Conversion from "Fraunhofer IPA Bin-Picking dataset" gt format (csv) to Sileane
dataset gt format (json)
Run script via:
    python <file> --path=/<path>/<to>/<workpiece_folder>/ --projectionType='o'|'p'

@author: Christian Landgraf

References:
    [1]  R. Bregier, F. Devernay, L. Leyrit, and J. L. Crowley, “Symmetry
         aware evaluation of 3d object detection and pose estimation in scenes
         of  many  parts  in  bulk,”  in The IEEE International Conference on
         Computer Vision (ICCV), Venice, Italy, Oct. 2017, pp. 2209–2218
"""

import glob
import os
import pandas as pd
import numpy as np

def convert_csv2json_gt(path, projectionType):
    """
    Converts our ground truth CSV format to the format of [1]
    """
    
    gt_path = os.path.join(path, "gt", "*", "*.csv")
    
    for file in glob.iglob(gt_path, recursive=True):
        df = pd.read_csv(file)
        if not df.empty:
            df = df.drop(columns=['id','class'])
        
        # save rotation axes
        tmp_df = pd.DataFrame()
        tmp_df['rot_x'] = df.iloc[:,3:10:3].values.tolist()
        tmp_df['rot_y'] = df.iloc[:,4:11:3].values.tolist()
        tmp_df['rot_z'] = df.iloc[:,5:12:3].values.tolist()    
        
        # combine single values in one column
        new_df = pd.DataFrame()
        new_df['R'] = tmp_df[:].values.tolist()
        new_df['segmentation_id'] = range(len(df))
        new_df['occlusion_rate'] = 1 - np.minimum(df['visibility_score_'+projectionType], [1.0]*len(df))
        new_df['t'] = df.iloc[:,0:3].values.tolist()
        
        # do not write bin position
        new_df.iloc[1:].to_json(os.path.splitext(file)[0]+".json",'records')



if __name__ == '__main__':    
    import argparse
    
    parser = argparse.ArgumentParser(description="This script converts our ground truth CSV format to the format of [1].")
    parser.add_argument('--path',default='./SileaneBunny/' ,
                        help="Path to workpiece")
    parser.add_argument('--projectionType', default='p', choices=['o','p'],
                        help = "orthogonal (o) or perspective (p) projection")
    
    args = parser.parse_args()

    convert_csv2json_gt(args.path, args.projectionType)
