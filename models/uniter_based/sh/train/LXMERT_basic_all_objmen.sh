#!/bin/bash
#SBATCH --job-name=LXMERT_basic_all_objmen       # 任务名
#SBATCH --nodes=1                   # 这里不用动 多节点脚本请查官方文档
#SBATCH --ntasks=1                  # 这里不用动 多任务脚本请查官方文档
#SBATCH --cpus-per-task=8           # 要几块CPU (一般4块就够用了)
#SBATCH --mem=256GB                 # 最大内存
#SBATCH --time=24:00:00             # 运行时间上限
#SBATCH --mail-type=END             # ALL / END
#SBATCH --mail-user=yh2689@nyu.edu  # 结束之后给哪里发邮件
#SBATCH --output=%x%A.out         # 正常输出写入的文件
#SBATCH --error=%x%A.err          # 报错信息写入的文件
#SBATCH --gres=gpu:1                # 需要几块GPU (同时最多8块)
#SBATCH -p aquila                   # 有GPU的partition
#SBATCH --nodelist=agpu7

module purge                        # 清除所有已加载的模块
module load anaconda3 cuda/11.1.1              # 加载anaconda (load virtual env for training)

nvidia-smi
nvcc --version
cd /gpfsnyu/scratch/yh2689/MMCoref/MMCoref_cleaned     # 切到程序目录

echo "START"               # 输出起始信息
source deactivate
source /gpfsnyu/packages/anaconda3/5.2.0/bin/activate wilson          # 调用 virtual env
python -u train.py \
    --NAME LXMERT_basic_all_objmen \
    --MODEL LXMERT \
    --obj_id True \
    --vis_feats_clip True \
    --vis_feats_rcnn True \
    --pos True \
    --scene_seg True \
    --obj_embs_bert True \
    --obj_embs_sbert True \
    --kb_id_bert False \
    --kb_id_sbert False \
    --obj_men True
echo "FINISH"                       # 输出起始信息
