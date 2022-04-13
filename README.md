### FPCC: Fast Point Cloud Clustering based Instance Segmentation for Industrial Bin-picking [<a href="https://arxiv.org/pdf/2012.14618.pdf">Arxiv</a>]
### Citation
If you find our work useful in your research, please consider citing:
	@article{XU2022,
	title = {FPCC: Fast Point Cloud Clustering-based Instance Segmentation for Industrial Bin-picking},
	journal = {Neurocomputing},
	year = {2022},
	issn = {0925-2312},
	author = {Yajun Xu and Shogo Arai and Diyi Liu and Fangzhou Lin and Kazuhiro Kosuge},
	}
### Dependencies
- `tensorflow` (1.13.1)
- `h5py`
### Data generation
Thanks for [<a href="https://github.com/waiyc">waiyc</a>] for providing  a [<a href="https://github.com/waiyc/Bin-Picking-Dataset-Generation">script</a>] to generate synthetic data by Pybullet.
The dataset is generate and recorded base on the steps mentioned in IPA Dataset.

### Training & Testing 


```bash
python fpcc_train.py 
```
```bash
python fpcc_test.py
```
### Evaluation metric
We use the code (eval_iou_accuracy.py) provided by [<a href="https://github.com/WXinlong/ASIS">ASIS</a>] to calculate precision and recall.

### Trained weights
We uploaded trained weights for the ring, gear, and part A. You can use these weights to test the performance of FPCC directly and debug. 
Please note that these weights are not the ones used in our paper.

### XA Dataset 
XA dataset can be downloaded [<a href="https://drive.google.com/drive/folders/1KCDS8_ZHxav5NZKhBzgEX4srf5xg7vW0?usp=sharing">here</a>].
Please cite this paper or "FPCC" if you want to use XA dataset in your work,

	@ARTICLE{9025047,
	 author={Xu, Yajun and Arai, Shogo and Tokuda, Fuyuki and Kosuge, Kazuhiro},
	 journal={IEEE Access},
	 title={A Convolutional Neural Network for Point Cloud Instance Segmentation in Cluttered Scene Trained by Synthetic Data Without Color},
	 year={2020},
	 volume={8},
	 number={},
	 pages={70262-70269},
	 doi={10.1109/ACCESS.2020.2978506}
	}
### IPA Dataset 
The information about IPA dataset can be found [<a href="https://www.bin-picking.ai/">here</a>].
The detail about IPA dataset can be downloaded [<a href="https://arxiv.org/abs/1912.12125">here</a>].
Dataset is available at [<a href="https://owncloud.fraunhofer.de/index.php/s/AacICuOWQVWDDfP?path=%2F">here</a>].


We only uploaded a part of the IPA dataset in the "datas" folder.
（1~3 col is the coordinates, 4th col is the visible score defined by the IPA dataset, and the last column (5th) is the instance label.）
Use the following scripts for generating h5 files for traing.
```bash
python convert_csv2json_annotation.py
python IPA_image2pc.py
python generate_ipa_center_h5.py
python generate_file_list.py
```

### Acknowledgemets

This project is built upon [<a href="https://github.com/charlesq34/pointnet">PointNet</a>], [<a href="https://github.com/laughtervv/SGPN">SGPN</a>] and [<a href="https://github.com/WangYueFt/dgcnn">DGCNN</a>]

### Others
If you have any questions or advice, please feel free to contact me at the address below.
baalxyj@gmail.com
