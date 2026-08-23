source ~/.bashrc
cd ~/llm_attacks
conda activate torchenv

squeue -u $USER | awk '{print $1}' | tail -n+2 | xargs scancel

srun -A galileo -p galileo  -c 4 --gres=gpu:1 python -m src.train_model.train_model \
   --dataset-setup attack_toxic_attack_gemma3_12b_it_lm_target_llama_31_8B_instruct_judge_detoxify\
    --dataset-name dataset_toxicity \
    --acquisition-strategy naive --last-round-epochs 500\
   --n-seed 3600 --epochs 2 --device cuda:0 --total-budget 10 --acquire-full-time 0 --data-type real &


srun -A galileo -p galileo  -c 4 --gres=gpu:1 python -m src.train_model.train_model \
   --dataset-setup attack_default_attack_gemma3_12b_it_lm_target_llama_31_8B_instruct_judge_llm-judge_gemma3_12b_it \
    --dataset-name dataset_red_team \
    --acquisition-strategy naive --last-round-epochs 500\
   --n-seed 3600 --epochs 2 --device cuda:0 --total-budget 10 --acquire-full-time 0 --data-type real &

attack_autoif_helper_qwen25_14b_instruct_lm_target_gemma3_4b_it_judge_autoif
srun -A galileo -p galileo  -c 4 --gres=gpu:1 python -m src.train_model.train_model \
   --dataset-setup attack_autoif_helper_qwen25_14b_instruct_lm_target_gemma3_4b_it_judge_autoif \
    --dataset-name dataset_autoif \
    --acquisition-strategy naive --last-round-epochs 500\
   --n-seed 3600 --epochs 2 --device cuda:0 --total-budget 10 --acquire-full-time 0 --data-type real &

srun -A galileo -p galileo  -c 4 --gres=gpu:1 python -m src.train_model.train_model \
   --dataset-setup attack_autoif_helper_qwen25_14b_instruct_lm_target_mini_phi_4_instruct_judge_autoif \
    --dataset-name dataset_autoif  --acquisition-strategy naive --last-round-epochs 500 \
   --n-seed 3600 --epochs 2 --device cuda:0 --total-budget 10 --acquire-full-time 0 --data-type real &

srun -A galileo -p galileo  -c 4 --gres=gpu:1 python -m src.train_model.train_model \
   --dataset-setup attack_hallucination_attack_qwen25_14b_instruct_lm_target_gemma3_4b_it_judge_llm-judge_qwen25_14b_instruct  \
    --dataset-name dataset_hallucination \
    --acquisition-strategy naive --last-round-epochs 500\
   --n-seed 3600 --epochs 2 --device cuda:0 --total-budget 10 --acquire-full-time 0 --data-type real &




srun -A galileo -p galileo -c4 --gres=gpu:1   --mem=20G \
 --exclude=$(sinfo -N -h -o "%n %G" | awk '$2 !~ /A100|A40|A6000|6000ADA|L40|L4|A4000/ {print $1}' \
  | paste -sd, -) -J plsNoKil python -m alg_stuff.train_model --acquisition-strategy naive --last-round-epochs 500\
   --n-seed 3600 --epochs 2 --device cuda:0 --total-budget 10 --acquire-full-time 0 --data-type real\
   --dataset-setup attack_toxic_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_detoxify\
    --dataset-name dataset_toxicity


setups=(
'attack_toxic_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_detoxify'\
 'attack_toxic_attack_qwen25_14b_instruct_lm_target_mini_phi_4_instruct_judge_detoxify'\
 'attack_toxic_attack_qwen25_14b_instruct_lm_target_llama_31_8B_instruct_judge_detoxify'\
 'attack_toxic_attack_qwen25_14b_instruct_lm_target_gemma3_4b_it_judge_detoxify'\
 )

for setup in "${setups[@]}"; do
  srun -p public,ash,nlp,dym,galileo,bml,tdk,espresso,euler,newton,ran  -c4 --gres=gpu:1   --mem=20G \
   --exclude=$(sinfo -N -h -o "%n %G" | awk '$2 !~ /A100|A40|A6000|6000ADA|L40|L4|A4000/ {print $1}' \
    | paste -sd, -) -J plsNoKil python -m alg_stuff.train_model --acquisition-strategy naive --last-round-epochs 500\
     --n-seed 3600 --epochs 2 --device cuda:0 --total-budget 10 --acquire-full-time 0 --data-type real\
     --dataset-setup "$setup"\
      --dataset-name dataset_toxicity&
done


setups=(
 'attack_default_attack_qwen25_14b_instruct_lm_target_mini_phi_4_instruct_judge_llm-judge_qwen25_14b_instruct'\
 'attack_default_attack_qwen25_14b_instruct_lm_target_llama_31_8B_instruct_judge_llm-judge_qwen25_14b_instruct'\
 'attack_default_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_llm-judge_qwen25_14b_instruct'\
 'attack_default_attack_qwen25_14b_instruct_lm_target_gemma3_4b_it_judge_llm-judge_qwen25_14b_instruct'\
 'attack_default_attack_qwen25_14b_instruct_lm_target_mini_phi_4_instruct_judge_llama_guard'\
 'attack_default_attack_qwen25_14b_instruct_lm_target_llama_31_8B_instruct_judge_llama_guard'\
 'attack_default_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_llama_guard'\
 )

for setup in "${setups[@]}"; do
  srun  -p galileo -A galileo  -c4 --gres=gpu:1   --mem=20G \
   --exclude=$(sinfo -N -h -o "%n %G" | awk '$2 !~ /A100|A40|A6000|6000ADA|L40|L4|A4000/ {print $1}' \
    | paste -sd, -) -J plsNoKil python -m alg_stuff.train_model --acquisition-strategy naive --last-round-epochs 500\
     --n-seed 3600 --epochs 2 --device cuda:0 --total-budget 10 --acquire-full-time 0 --data-type real\
     --dataset-setup "$setup"\
      --dataset-name dataset_red_team &
  done

setups=(
'attack_hallucination_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_llm-judge_qwen25_14b_instruct'
'attack_hallucination_attack_qwen25_14b_instruct_lm_target_llama_31_8B_instruct_judge_llm-judge_qwen25_14b_instruct'
'attack_hallucination_attack_qwen25_14b_instruct_lm_target_mini_phi_4_instruct_judge_llm-judge_qwen25_14b_instruct'
 )
for setup in "${setups[@]}"; do
  srun   -p galileo -A galileo  -c4 --gres=gpu:1   --mem=20G \
   --exclude=$(sinfo -N -h -o "%n %G" | awk '$2 !~ /A100|A40|A6000|6000ADA|L40|L4|A4000/ {print $1}' \
    | paste -sd , -),plato[1-2],ran-mashawsha,newton3 -J plsNoKil python -m alg_stuff.train_model --acquisition-strategy naive --last-round-epochs 500\
     --n-seed 3600 --epochs 2 --device cuda:0 --total-budget 10 --acquire-full-time 0 --data-type real\
     --dataset-setup "$setup"\
      --dataset-name dataset_hallucination3 &
  done


setups=(
'attack_autoif_helper_qwen25_14b_instruct_lm_target_llama_31_8B_instruct_judge_autoif'
'attack_autoif_helper_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_autoif'
 )
for setup in "${setups[@]}"; do
  srun   -p galileo -A galileo  -c4 --gres=gpu:1   --mem=20G \
   --exclude=$(sinfo -N -h -o "%n %G" | awk '$2 !~ /A100|A40|A6000|6000ADA|L40|L4|A4000/ {print $1}' \
    | paste -sd , -),plato[1-2],ran-mashawsha,newton3 -J plsNoKil python -m alg_stuff.train_model --acquisition-strategy naive --last-round-epochs 500\
     --n-seed 3600 --epochs 2 --device cuda:0 --total-budget 10 --acquire-full-time 0 --data-type real\
     --dataset-setup "$setup"\
      --dataset-name dataset_autoif &
  done
