#!/bin/bash
# run with 4 random seeds for each annotator with save hyperparameters

cd ..
source venv/bin/activate

GPUs=() # set gpus
seed=0
budget=0.5
annotators_1=("1,2,6,7,11,12,13,15,17,18,21,22-\
4,6,7,8,9,10,11,14,16,17,22,23-0,1,2,3,4,5,6,10,15,16,21,22-\
3,7,8,9,10,11,13,16,17,19,21,23-\
2,3,4,5,6,8,13,15,16,19,21,22-1,2,3,4,6,9,10,11,14,18,20,23-0,3,4,5,7,8,12,13,16,17,19,20-\
1,2,3,4,5,6,13,14,15,16,19,22-\
0,1,2,6,9,10,12,13,18,19,21,22-\
0,2,3,7,10,12,14,15,17,21,22,23"

"2,6,7,8,9,11,12,13,17,21,22,23-0,2,5,6,10,14,15,16,17,18,20,23-\
0,3,4,5,6,7,8,10,13,16,20,21-\
0,1,5,8,12,14,16,17,19,20,21,22-\
2,4,5,7,10,12,13,14,17,18,20,23-1,3,7,9,11,13,17,18,19,20,22,23-\
0,1,3,5,6,9,10,11,12,14,17,20-\
0,1,3,4,6,8,11,13,14,16,17,23-\
0,7,8,9,12,13,16,17,20,21,22,23-\
2,3,4,7,8,11,12,13,16,18,20,22")

for ann_index in  "${!annotators_1[@]}" ; do
    
    tasks_1=${annotators_1[$ann_index]}
    # echo tasks_1
    # tasks_5=${annotators_5[$ann_index]}
    
    # Get current gpu
    gpu_index=$(($ann_index % ${#GPUs[@]}))
    gpu=${GPUs[$gpu_index]}
    
    echo "Running model for tasks: $tasks_1, gpu: $gpu"
    
    SESSION_NAME="${gpu}_"
    
    screen -dmS "$SESSION_NAME" bash -c "CUDA_VISIBLE_DEVICES=$gpu python mtl_main.py \
                                                    --dataset "mfrc" \
                                                    --label "Moral" \
                                                    --train_batch_size 64 \
                                                    --budget $budget \
                                                    --mtl_tasks "$tasks_1" \
                                                    --run_sweep \
                                                    --seed $seed ;
                                        "
done

