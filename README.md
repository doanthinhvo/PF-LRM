<div align="center">
  
# PF-LRM: Pose-Free Large Reconstruction Model for Joint Pose and Shape Prediction

<a href="https://arxiv.org/abs/2311.12024"><img src="https://img.shields.io/badge/ArXiv-2404.07191-brightgreen"></a> 
</div>

---

This repo is the unofficial implementation of [PF-LRM](https://arxiv.org/abs/2311.12024), Pose-Free Large Reconstruction Model for reconstructing a 3D object from a few unposed images even with little visual overlap, while simultaneously estimating the relative camera poses in ~1.3 seconds on a single A100 GPU.


# Features and Todo List
- [ ] Release inference code. 
- [ ] Support for running on Google Colabs.
- [ ] Release model weights.

# Dependencies and Installation

We recommend using `Python>=3.10`, `PyTorch>=2.1.0`, and `CUDA>=12.1`.
```bash
conda create --name pflrm python=3.10
conda activate pflrm
pip install -U pip

# Ensure Ninja is installed
conda install Ninja

# Install the correct version of CUDA
conda install cuda -c nvidia/label/cuda-12.1.0

# Install PyTorch and xformers
# You may need to install another xformers version if you use a different PyTorch version
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.22.post7

# For Linux users: Install Triton 
pip install triton

# For Windows users: Use the prebuilt version of Triton provided here:
pip install https://huggingface.co/r4ziel/xformers_pre_built/resolve/main/triton-2.0.0-cp310-cp310-win_amd64.whl

# Install other requirements
pip install -r requirements.txt
```

# Training

We provide our training code to facilitate future research. But we cannot provide the training dataset due to its size. Please refer to our [dataloader](src/data/objaverse.py) for more details.

To train the PF-LRM, please run:
```bash
python train.py --base configs/pf-lrm-small-train.yaml --gpus 0,1,2,3 --num_nodes 1
```

If you find our work useful for your research or applications, please cite using this BibTeX:

```@article{Wang2023PFLRMPL,
  title={PF-LRM: Pose-Free Large Reconstruction Model for Joint Pose and Shape Prediction},
  author={Peng Wang and Hao Tan and Sai Bi and Yinghao Xu and Fujun Luan and Kalyan Sunkavalli and Wenping Wang and Zexiang Xu and Kai Zhang},
  journal={ArXiv},
  year={2023},
  volume={abs/2311.12024},
  url={https://api.semanticscholar.org/CorpusID:265295290}}
```

# Acknowledgements

This repo is built upon the code of InstantMesh and we thank the developers for their great works and efforts to release source code. Furthermore, a special "thank you" to PF-LRM's authors for publishing such an amazing work.