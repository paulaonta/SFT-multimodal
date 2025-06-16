#!/bin/bash
#SBATCH --partition=hitz-exclusive
#SBATCH --account=hitz-exclusive
#SBATCH --job-name=lm_eval_imgOnly_dev_no_ft
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=50GB
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-user=paula.ontalvilla@ehu.eus 
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --output=../results/.slurm/imgOnly_dev_no_ft/lm_eval_imgOnly_dev_reka_flash.out
#SBATCH --error=../results/.slurm/imgOnly_dev_no_ft/lm_eval_imgOnly_dev_reka_flash.err

# activate virtual environment
module load Python

source /scratch/pontalvillla/multimodal/qwen_env/bin/activate

path="/home/pontalvillla/multimodal/multimodal/MATH500/data/500_MATH_train/"
textData_path="/home/pontalvillla/multimodal/multimodal/MATH500/data/dev_MATH500.json"
script_path="/scratch/pontalvillla/multimodal/general_scripts/"

srun python3 "${script_path}eval_imgOnly_mlm.py" \
    --model_name "unsloth/Llama-3.2-11B-Vision-Instruct" \
    --datasetText_name $textData_path \
    --datasetImg_path $path\
    --prompt "  Solve the math problem of the image step by step. The last line of your response should be of the form Answer: \$ANSWER (without quotes) where \$ANSWER is the answer to the problem. Remember to put your answer on its own line after "Answer:", and you do not need to use a \\\\boxed command. "\
    --metric exact_match \
    --max_length 2000

