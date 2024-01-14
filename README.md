<p>The official PyTorch implementation of "SOSNet: Real-Time Small Object Segmentation via Hierarchical Decoding and Example Mining"</p>

<div style="display:flex;justify-content:center; text-align:center"> 
  <p style="">Improvement</p>
  <img src="https://github.com/StuLiu/SOSNet/blob/master/assests/improve.png" width="600px" style="">
  <p style="">Method overview</p>
  <img src="https://github.com/StuLiu/SOSNet/blob/master/assests/overview.png" width="600px" style="">
</div>

## <div align="center">Usage</div>

<details open>
  <summary><strong>Installation</strong></summary>

* python >= 3.6
* torch >= 1.8.1
* torchvision >= 0.9.1

Then, clone the repo and install the project with:

```bash
$ git clone https://github.com/StuLiu/sosnet
$ cd sosnet
$ pip install -e .
```

</details>

<br>
<details>
  <summary><strong>Configuration</strong> (click to expand)</summary>

Create a configuration file in `configs`. Sample configuration for segformer dataset can be found [here](configs/SegFormer). Then edit the fields you think if it is needed. This configuration file is needed for all of training, evaluation and prediction scripts.

</details>

<br>
<details>
  <summary><strong>Training</strong> (click to expand)</summary>

Prepare data: 
* [Camvid 360*480](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid)
* [Vaihingen](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx)
* [UAVid2020](https://uavid.nl/)

Download pretrained module
* [mit-b0 and mobilenetV3-large](https://drive.google.com/drive/folders/1CE9FKxF0TidCUqlc9JmtBLYnKWTt8z71?usp=sharing)


To train with a single GPU:

```bash
$ python tools/train_sosnet.py --cfg configs/CONFIG_FILE.yaml --hier 1 --soem 1
```

[//]: # ()
[//]: # (To train with multiple gpus, set `DDP` field in config file to `true` and run as follows:)

[//]: # ()
[//]: # (```bash)

[//]: # ($ python -m torch.distributed.launch --nproc_per_node=2 --use_env tools/train.py --cfg configs/<CONFIG_FILE_NAME>.yaml)

[//]: # (```)

</details>

<br>
<details>
  <summary><strong>Evaluation</strong> (click to expand)</summary>

Make sure to set `MODEL_PATH` of the configuration file to your trained model directory.

```bash
$ python tools/val.py --cfg configs/<CONFIG_FILE_NAME>.yaml
```

To evaluate with multi-scale and flip, change `ENABLE` field in `MSF` to `true` and run the same command as above.

</details>

<br>
<details open>
  <summary><strong>Inference</strong></summary>

To make an inference, edit the parameters of the config file from below.
* Change `MODEL` >> `NAME` and `BACKBONE` to your desired pretrained model.
* Change `DATASET` >> `NAME` to the dataset name depending on the pretrained model.
* Set `TEST` >> `MODEL_PATH` to pretrained weights of the testing model.
* Change `TEST` >> `FILE` to the file or image folder path you want to test.
* Testing results will be saved in `SAVE_DIR`.

```bash
## example using ade20k pretrained models
$ python tools/infer.py --cfg configs/CONFIGFILE.yaml
$ python tools/infer_single.py --img_path demo/camvid_0.png --cfg configs/segformer/camvid_mitb0.yaml
```

</details>

<b>Cite/Reference</b>
<p>If you find SOSNet useful in your research, please consider citing:</p>

```
@ARTICLE{10359121,
  author={Liu, Wang and Kang, Xudong and Duan, Puhong and Xie, Zhuojun and Wei, Xiaohui and Li, Shutao},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={SOSNet: Real-Time Small Object Segmentation via Hierarchical Decoding and Example Mining}, 
  year={2023},
  volume={},
  number={},
  pages={1-13},
  doi={10.1109/TNNLS.2023.3338732}}

```

<b>Acknowledge</b>
<p>This project is based on the repository 'semantic-segmentation'.</p>

* [semantic-segmentation](https://github.com/sithu31296/semantic-segmentation)
