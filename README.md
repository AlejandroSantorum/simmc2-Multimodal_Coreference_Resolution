# (SIMMC 2.0) Challenge 2021: Multimodal Coreference Resolution task
The second Situated Interactive MultiModal Conversations (SIMMC 2.0) Challenge 2021. Project focused on the second task: Multimodal Coreference Resolution.

The [official GitHub Repository](https://github.com/facebookresearch/simmc2) of the challenge is published by the [Meta Research](https://github.com/facebookresearch) team.

The Multimodal Coreference Resolution task is one of the proposed tracks of the Tenth Dialog System Technology Challenge [DSTC10](https://sites.google.com/dstc.community/dstc10/home).

Clone this repository to download the dataset and the code:
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


----------------------------------------------
## Download the Dataset
The dataset is hosted in [this](https://github.com/AlejandroSantorum/simmc2-Multimodal_Coreference_Resolution) GitHub Repository with [Git LFS](https://git-lfs.github.com/) and in the [challenge repository](https://github.com/facebookresearch/simmc2/tree/main/data). Make sure to install and update Git LFS before cloning the repository:
```bash
$ git lfs install
```
