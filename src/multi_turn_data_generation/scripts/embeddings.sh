
srun -A galileo -p galileo  -c 4 --gres=gpu:1 python -m src.multi_turn_data_generation.embedding --idx-start 0 --idx-end 2000 &
srun -A galileo -p galileo  -c 4 --gres=gpu:1 python -m src.multi_turn_data_generation.embedding --idx-start 2000 --idx-end 4000 &
srun -A galileo -p galileo  -c 4 --gres=gpu:1 python -m src.multi_turn_data_generation.embedding --idx-start 4000 --idx-end 6000 &
srun -A galileo -p galileo  -c 4 --gres=gpu:1 python -m src.multi_turn_data_generation.embedding --idx-start 6000 --idx-end 8000 &
srun -A galileo -p galileo  -c 4 --gres=gpu:1 python -m src.multi_turn_data_generation.embedding --idx-start 8000 --idx-end 10000 &

tar -czf embeddings.tar.gz results/embedding/dataset_red_team/