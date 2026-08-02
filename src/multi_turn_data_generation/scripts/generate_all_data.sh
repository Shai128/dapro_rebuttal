#!/bin/bash
source ~/.bashrc
cd ~/dapro_rebuttal
conda activate torchenv
squeue -u $USER | awk '{print $1}' | tail -n+2 | xargs scancel
git pull

# Configuration arrays
#dataset_names=('toxicity' 'red_team' 'hallucination' 'autoif')
#attacker_models=('qwen25-14b-instruct')
#target_models=('qwen25-14b-instruct' 'llama-3.1-8B-instruct' 'mini_phi_4_instruct' 'gemma3_4b_it')


dataset_names=('hallucination')
attacker_models=('gemma3_12b_it')
target_models=('llama-3.1-8B-instruct')
exclude_list=$(sinfo -N -h -o "%n %G" | awk '$2 !~ /A100|A40|A6000|6000ADA|L40|L4|A4000/ {print $1}' | paste -sd, -)
export PYTHONPATH="src/multi_turn_data_generation:${PYTHONPATH:-}"
# Iterate through combinations
for dataset in "${dataset_names[@]}"; do

    # Define the list of judges based on the dataset
    case "$dataset" in
        "toxicity")
            judges=("detoxify")
            ;;
        "red_team")
            judges=("llm-judge" "llama-guard")
            ;;
        "hallucination" | "autoif")
            judges=("llm-judge")
            ;;
        *)
            echo "Unknown dataset: $dataset. Skipping."
            continue
            ;;
    esac

    for judge in "${judges[@]}"; do
        for attacker in "${attacker_models[@]}"; do
            for target in "${target_models[@]}"; do

                pids=()

                for ((i = 0; i < 10; i++)); do
                    index_start=$((i * 1000))
                    index_end=$(((i + 1) * 1000))

                    echo "Starting process $i: dataset=$dataset, judge=$judge, attacker=$attacker, target=$target, indices=$index_start-$index_end"
#                   srun -p public,ash,nlp,dym,galileo,bml,tdk,espresso,euler,newton,ran -c4 --gres=gpu:1 --mem=20G  \
#                    --exclude="$exclude_list" -J plsNoKil \

#                   srun -A galileo -p galileo  -c 4 --gres=gpu:1 \

                    srun -A galileo -p galileo  -c 4 --gres=gpu:1\
                    --exclude="$exclude_list" -J plsNoKil \
                       python -m src.multi_turn_data_generation.main \
                            --data-index-start "$index_start"  --data-index-end "$index_end" \
                            --target-model "$target"  --n-iterations 200 \
                            --dataset-name "$dataset"  --judge-model "$judge" \
                            --batch-size 1000  --max-n-attack-attempts 20 \
                            --attack-model "$attacker" &

                    pids+=("$!")
                done

                # Wait for all 10 processes for this configuration
                failed=0

                for pid in "${pids[@]}"; do
                    if ! wait "$pid"; then
                        echo "Process $pid failed."
                        failed=1
                    fi
                done

                if ((failed)); then
                    echo "One or more processes failed: dataset=$dataset, judge=$judge, attacker=$attacker, target=$target"
                else
                    echo "Finished all index ranges: dataset=$dataset, judge=$judge, attacker=$attacker, target=$target"
                fi

                echo "-----------------------------------"
            done
        done
    done
done