#!/bin/bash
# run with 4 random seeds for each annotator with save hyperparameters

cd ..
source venv/bin/activate


budget=(0.5 0.66 0.83 1)
GPUs=() 
seed=0
# annotators_1=("0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23")

# Loop over the first half of tasks and run in parallel
for b_index in "${!budget[@]}" ; do
    tasks_1=${annotators_1[$ann_index]}
    b=${budget[$b_index]}
    # Get current gpu
    gpu_index=$(($b_index % ${#GPUs[@]}))
    gpu=${GPUs[$gpu_index]}
    
    echo "Running model for tasks: $tasks_1, gpu: $gpu"
    
    SESSION_NAME="brexit_gpu_${gpu}_seed_${seed}_budget_${b}_alltasks"

    screen -dmS "$SESSION_NAME" bash -c "CUDA_VISIBLE_DEVICES=$gpu python mtl_test_model.py \
                                                --seed $seed \
                                                --split "test" \
                                                --budget $b ;  
                                        "
done
