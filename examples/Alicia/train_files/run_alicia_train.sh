###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=ABot_M0
freeze_module_list=""
base_vlm=path_to_your_base_vlm_checkpoint
config_yaml=./examples/Alicia/train_files/ABot_alicia.yaml
datasets_root=/home/kongqingwei/data
dataset_name=recore_dataset_two_box_location_offical
pretrain_ckpt=path_to_your_pretrain_ckpt
run_root_dir=./results/Checkpoints
run_id=alicia_ft_abot_m0
num_processes=1
# === End of environment variable configuration ===
###########################################################################################

export WANDB_MODE=offline
export WANDB_DISABLED=true
export CUDA_HOME=/usr/local/cuda-12
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
export PATH="$HOME/.local/bin:$PATH"
export HF_ENDPOINT=https://hf-mirror.com

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

accelerate launch \
  --config_file ABot/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${num_processes} \
  ABot/training/train.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${datasets_root} \
  --datasets.vla_data.dataset_name ${dataset_name} \
  --trainer.pretrained_checkpoint ${pretrain_ckpt} \
  --trainer.reload_modules qwen_vl_interface,action_model \
  --trainer.freeze_modules ${freeze_module_list} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id}
