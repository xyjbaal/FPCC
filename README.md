### FPCC: Fast Point Cloud Clustering-based Instance Segmentation for Industrial Bin-picking [<a href="https://arxiv.org/pdf/2012.14618.pdf">Arxiv</a>]

![NMS](https://github.com/xyjbaal/FPCC/blob/main/nms.png)

## Other Implementation
* [pytorch-fpcc](https://github.com/xyjbaal/fpcc_pytorch)
### Citation
If you find our work useful in your research, please consider citing:

	@article{XU2022255,
	title = {FPCC: Fast point cloud clustering-based instance segmentation for industrial bin-picking},
	journal = {Neurocomputing},
	volume = {494},
	pages = {255-268},
	year = {2022},
	issn = {0925-2312},
	doi = {https://doi.org/10.1016/j.neucom.2022.04.023},
	url = {https://www.sciencedirect.com/science/article/pii/S0925231222003915},
	author = {Yajun Xu and Shogo Arai and Diyi Liu and Fangzhou Lin and Kazuhiro Kosuge},
	keywords = {Bin-picking, 3D Point Cloud, Instance segmentaion, Deep Learning},
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
We use the code provided by [<a href="https://github.com/WXinlong/ASIS">ASIS</a>] to calculate precision and recall.


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
The paper about IPA dataset can be downloaded [<a href="https://arxiv.org/abs/1912.12125">here</a>].
The author of IPA did not provide a public link to download the data set, so maybe you need to register first.

We only uploaded part of the IPA dataset in the "datas" folder.
Use the following scripts for generating h5 files for traing.
```bash
python convert_csv2json_annotation.py
python IPA_image2pc.py
python generate_ipa_center_h5.py
python generate_file_list.py
```

### Acknowledgemets

This project is built upon [<a href="https://github.com/charlesq34/pointnet">PointNet</a>], [<a href="https://github.com/charlesq34/pointnet2">PointNet++</a>], 
[<a href="https://github.com/laughtervv/SGPN">SGPN</a>] and [<a href="https://github.com/WangYueFt/dgcnn">DGCNN</a>]

### Others
The program is not very beautiful.
If you have any questions, please feel free to contact me at the address below.
baalxyj@gmail.com
