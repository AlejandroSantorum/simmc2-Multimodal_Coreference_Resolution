# (SIMMC 2.0) Challenge 2021: Multimodal Coreference Resolution task
The second Situated Interactive MultiModal Conversations (SIMMC 2.0) Challenge 2021. Project focused on the second task: Multimodal Coreference Resolution.

The [official GitHub Repository](https://github.com/facebookresearch/simmc2) of the challenge is published by [Meta Research](https://github.com/facebookresearch) team.

The Multimodal Coreference Resolution task is one of the proposed tracks of the Tenth Dialog System Technology Challenge [DSTC10](https://sites.google.com/dstc.community/dstc10/home).

Clone this repository to download the code and experiments:
```bash
$ git clone https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution.git
```

----------------------------------------------
## Configuration (recommended)

Using a virtual environment is recommended to minimize the chance of conflicts.

#### Setting up a virtual environment: `venv`

In the cloned repository folder, create a virtual environment (you can change the name "env").
```bash
python3 -m venv env
```

Activate the environment "env".
```bash
source env/bin/activate
```

Install requirements using `pip`.
```bash
pip install -r requirements.txt
```
You may need to use `pip3`.
```bash
pip3 install -r requirements.txt
```

#### Python requirements
The requirements to run the code are listed in `requirements.txt`:
* [numpy](https://github.com/numpy/numpy) - The fundamental package for scientific computing with Python
* [torch](https://pytorch.org/) - Optimized tensor Python library for deep learning using GPUs and CPUs.
* [transformers](https://huggingface.co/docs/transformers/index) - State-of-the-art Machine Learning library for PyTorch, TensorFlow and JAX.
* [tqdm](https://github.com/tqdm/tqdm) - Python library to make loops show a smart progress meter.
* [tensorboardX](https://pypi.org/project/tensorboardX/) - Python library to watch tensors flow without Tensorflow.


----------------------------------------------
## Download the Dataset
The dataset is hosted in [Meta's GitHub Repository](https://github.com/facebookresearch/simmc2) with [Git LFS](https://git-lfs.github.com/). The folder [data](https://github.com/facebookresearch/simmc2/tree/main/data) contains the whole dataset and the instructions to be downloaded.

Make sure to install and update Git LFS before cloning the repository:
```bash
$ git lfs install
```

```bash
$ git clone https://github.com/facebookresearch/simmc2.git
```

You may need to pull using Git LFS:
```bash
$ git lfs pull
```
