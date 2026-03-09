# tsslam
# TS-SLAM

## Installation

**a. Create a conda virtual environment and activate it.**

```bash
conda create --name transplat -y python=3.10.14
conda activate transplat
conda install -y pip
```

**b. Install PyTorch and torchvision.**

```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
# Recommended torch==2.1.2
```

**c. Install mmcv.**

```bash
pip install openmim
mim install mmcv==2.1.0
```

**d. Install other requirements.**
```bash
pip install -r requirements.txt
```

## Datasets Download
- TUM Dataset: [https://vision.in.tum.de/rgbd/dataset/](https://vision.in.tum.de/rgbd/dataset/)
- ETH3D SLAM Dataset: [https://www.eth3d.net/slam_datasets](https://www.eth3d.net/slam_datasets)
- Processed scnnnet Dataset: [https://pan.baidu.com/s/1g_6SBzuCodKoRZe42vUyhw?pwd=ppph](https://pan.baidu.com/s/1g_6SBzuCodKoRZe42vUyhw?pwd=ppph)
- Self-collected Dataset by Our Robot: [https://pan.baidu.com/s/1pX-Y3ckebDC8UsTVUqPpBw?pwd=23g9](https://pan.baidu.com/s/1pX-Y3ckebDC8UsTVUqPpBw?pwd=23g9)

## Training
```bash
python transformermodel/src/train.py +experiment=re10k \
checkpointing.load=./checkpoints/checkpoints.ckpt \
data_loader.train.batch_size=1\
dataset.view_sampler.num_context_views=2 \
wandb.mode=run\
wandb.name=train\
test.compute_scores=true 
```

## Testing
```bash
python slam.py --config configs/mono/tum/fr3_office.yaml
```

## Acknowledgements
This project is built upon and references the following open-source repositories:
- [https://github.com/xingyoujun/transplat](https://github.com/xingyoujun/transplat)
- [https://github.com/muskie82/MonoGS](https://github.com/muskie82/MonoGS)
- [https://github.com/DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)

