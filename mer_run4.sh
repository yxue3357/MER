seed=$1

ROT="--n_layers 3 --n_hiddens 100 --data_path data/ --save_path results/ --batch_size 1 --log_every 100 --samples_per_task 1000 --data_file mnist_rotations.pt"
PERM="--n_layers 3 --n_hiddens 100 --data_path data/ --save_path results/ --batch_size 1 --log_every 100 --samples_per_task 1000 --data_file mnist_permutations.pt"
MANY="--n_layers 3 --n_hiddens 100 --data_path data/ --save_path results/ --batch_size 1 --log_every 100 --samples_per_task 200 --data_file mnist_manypermutations.pt"
echo "Beginning ER (Algorithm 4) With 5120 Memories" "( seed =" $seed ")"
# python3 main.py $MANY --seed $seed --model eralg4_adv --lr 0.1 --memories 5120 --replay_batch_size 25 --adv_beta 0.5
python3 main.py $PERM --seed $seed --model eralg4_cv --lr 0.1 --memories 1024 --replay_batch_size 25 
python3 main.py $PERM --seed $seed --model eralg4 --lr 0.1 --memories 1024 --replay_batch_size 25 
