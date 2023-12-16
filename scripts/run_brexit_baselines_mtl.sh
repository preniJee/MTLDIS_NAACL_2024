#!/bin/bash
# run with 4 random seeds for each annotator with save hyperparameters

cd ..
source venv/bin/activate
seed=0 # set seed 1,2
GPUs=() # set gpus
budget=(0.5 0.66 0.83 1)
annotators_1=("Ann1,Ann2,Ann3,Ann4,Ann5,Ann6")

# Loop over the first half of tasks and run in parallel
for ann_index in  "${!annotators_1[@]}" ; do
    for b_index in "${!budget[@]}" ; do
        tasks_1=${annotators_1[$ann_index]}
        b=${budget[$b_index]}
        # Get current gpu
        gpu_index=$(($b_index % ${#GPUs[@]}))
        gpu=${GPUs[$gpu_index]}
        
        echo "Running model for tasks: $tasks_1, gpu: $gpu"
        
        SESSION_NAME="gpu_${gpu}_budget_${b}_tasks_${tasks_1}"
        
        screen -dmS "$SESSION_NAME" bash -c "CUDA_VISIBLE_DEVICES=$gpu python mtl_main.py \
                                                        --mtl_tasks "$tasks_1" \
                                                        --budget $b \
                                                        --run_sweep \
                                                        --baseline \
                                                        --seed  $seed ;
                                            "
    done
   
done