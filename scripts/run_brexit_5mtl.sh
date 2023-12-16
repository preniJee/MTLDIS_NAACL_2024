#!/bin/bash
# run with 4 random seeds for each annotator with save hyperparameters

cd ..
source venv/bin/activate

GPUs=()
budget=0.83
annotators_1=("Ann1,Ann2,Ann3,Ann4,Ann5" "Ann1,Ann2,Ann3,Ann4,Ann6" "Ann1,Ann2,Ann3,Ann5,Ann6" "Ann1,Ann2,Ann4,Ann5,Ann6" "Ann1,Ann3,Ann4,Ann5,Ann6" "Ann2,Ann3,Ann4,Ann5,Ann6")
seed=0

for ann_index in  "${!annotators_1[@]}" ; do
    
    tasks_1=${annotators_1[$ann_index]}
    
    # Get current gpu
    gpu_index=$(($ann_index % ${#GPUs[@]}))
    gpu=${GPUs[$gpu_index]}
    
    echo "Running model for tasks: $tasks_1, gpu: $gpu"
    
    SESSION_NAME="${gpu}_${tasks_1}"
    
    screen -dmS "$SESSION_NAME" bash -c "CUDA_VISIBLE_DEVICES=$gpu python mtl_main.py \
                                                    --mtl_tasks "$tasks_1" \
                                                    --budget $budget \
                                                    --run_sweep \
                                                    --seed $seed ;
                                        "
done