#!/bin/bash
# run with 4 random seeds for each annotator with save hyperparameters

cd ..
source venv/bin/activate

GPUs=() # set gpus
budget=0.5
seed=0 # 
annotators_1=("Ann1,Ann2,Ann3" "Ann1,Ann2,Ann4" "Ann1,Ann2,Ann5" "Ann1,Ann2,Ann6" "Ann1,Ann3,Ann4")
annotators_2=("Ann1,Ann3,Ann5" "Ann1,Ann3,Ann6" "Ann1,Ann4,Ann5" "Ann1,Ann4,Ann6" "Ann1,Ann5,Ann6")
annotators_3=("Ann2,Ann3,Ann4" "Ann2,Ann3,Ann5" "Ann2,Ann3,Ann6" "Ann2,Ann4,Ann5" "Ann2,Ann4,Ann6")
annotators_4=("Ann2,Ann5,Ann6" "Ann3,Ann4,Ann5" "Ann3,Ann4,Ann6" "Ann3,Ann5,Ann6" "Ann4,Ann5,Ann6")

for ann_index in  "${!annotators_1[@]}" ; do
    
    tasks_1=${annotators_1[$ann_index]}
    tasks_2=${annotators_2[$ann_index]}
    tasks_3=${annotators_3[$ann_index]}
    tasks_4=${annotators_4[$ann_index]}
    echo "heyy $tasks"
    # Get current gpu
    gpu_index=$(($ann_index % ${#GPUs[@]}))
    gpu=${GPUs[$gpu_index]}
    
    echo "Running model for tasks: $tasks_1, gpu: $gpu"
    
    SESSION_NAME="${gpu}_${tasks_1}"
    
    screen -dmS "$SESSION_NAME" bash -c "CUDA_VISIBLE_DEVICES=$gpu python mtl_main.py \
                                                    --mtl_tasks "$tasks_1" \
                                                    --run_sweep \
                                                    --budget $budget \
                                                    --seed $seed ;
                                        CUDA_VISIBLE_DEVICES=$gpu python mtl_main.py \
                                                    --mtl_tasks "$tasks_2" \
                                                    --run_sweep \
                                                    --budget $budget \
                                                    --seed $seed ;
                                        CUDA_VISIBLE_DEVICES=$gpu python mtl_main.py \
                                                    --mtl_tasks "$tasks_3" \
                                                    --run_sweep \
                                                    --budget $budget \
                                                    --seed $seed ;
                                        CUDA_VISIBLE_DEVICES=$gpu python mtl_main.py \
                                                    --mtl_tasks "$tasks_4" \
                                                    --run_sweep \
                                                    --budget $budget \
                                                    --seed $seed ; 
                                        "
done
