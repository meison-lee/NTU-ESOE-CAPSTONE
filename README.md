# NTU-ESOE-Capstone
NTU-ESOS-Capstone topics of Engineering Science and Ocean Engineering
## Abstract

This is an implementation of AI-based Face Anti-Spoofing(FAS) Detection. The goal of this project is to perform FAS detection on Rasberry Pi 4. We trained CDCN++ which was proposed in "***Searching Central Difference Convolutional Networks for Face Anti-Spoofing***" from scratch. The model was trained on OULU-NPU and SiW dataset.

## Demo
* #### facial depth map

https://user-images.githubusercontent.com/112916328/221432342-abcf6f0a-64c5-4c50-9437-fb532ceb6b6f.mp4

* #### Run on Rasberry Pi 4 
(kind of laggy due to the limited computation resources on Rasberry Pi)

https://user-images.githubusercontent.com/112916328/221432436-0f363229-836f-4db3-851a-f4c91d912ca4.mp4

* #### Run on my laptop

https://user-images.githubusercontent.com/112916328/221432965-774d3581-c049-4d62-885f-15678797e1d9.mp4

## Usage
- Weights
    ```bash
    ```
    Weights can be acquired here.

- Start detecting!
    ```bash
    $ python run.py --detector_path --model_weights
    ```
    Use `--detector_path` and `--model_weights` to assign paths.

## Citation
```
@inproceedings{yu2020searching,
    title={Searching Central Difference Convolutional Networks for Face Anti-Spoofing},
    author={Yu, Zitong and Zhao, Chenxu and Wang, Zezheng and Qin, Yunxiao and Su, Zhuo and Li, Xiaobai and Zhou, Feng and Zhao, Guoying},
    booktitle= {CVPR},
    year = {2020}
}

@inProceedings{feng2018prn,
  title     = {Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network},
  author    = {Yao Feng and Fan Wu and Xiaohu Shao and Yanfeng Wang and Xi Zhou},
  booktitle = {ECCV},
  year      = {2018}
}
31
```
