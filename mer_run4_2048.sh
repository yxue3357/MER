ROT="--n_layers 3 --n_hiddens 100 --data_path data/ --save_path results/ --batch_size 1 --log_every 100 --samples_per_task 1000 --data_file mnist_rotations.pt"
PERM="--n_layers 3 --n_hiddens 100 --data_path data/ --save_path results/ --batch_size 1 --log_every 100 --samples_per_task 1000 --data_file mnist_permutations.pt"
MANY="--n_layers 3 --n_hiddens 100 --data_path data/ --save_path results/ --batch_size 1 --log_every 100 --samples_per_task 200 --data_file mnist_manypermutations.pt"


echo "-----------------------------------------------"
echo "-----------------------------------------------"
python3 main.py $PERM --seed 90 --model eralg4_cv --lr 0.1 --memories 2048 --replay_batch_size 25 
python3 main.py $PERM --seed 90 --model eralg4 --lr 0.1 --memories 2048 --replay_batch_size 25 

echo "-----------------------------------------------"
echo "-----------------------------------------------"
python3 main.py $PERM --seed 95 --model eralg4_cv --lr 0.1 --memories 2048 --replay_batch_size 25 
python3 main.py $PERM --seed 95 --model eralg4 --lr 0.1 --memories 2048 --replay_batch_size 25 

echo "-----------------------------------------------"
echo "-----------------------------------------------"
python3 main.py $PERM --seed 100 --model eralg4_cv --lr 0.1 --memories 2048 --replay_batch_size 25 
python3 main.py $PERM --seed 100 --model eralg4 --lr 0.1 --memories 2048 --replay_batch_size 25 

echo "-----------------------------------------------"
echo "-----------------------------------------------"
python3 main.py $PERM --seed 105 --model eralg4_cv --lr 0.1 --memories 2048 --replay_batch_size 25 
python3 main.py $PERM --seed 105 --model eralg4 --lr 0.1 --memories 2048 --replay_batch_size 25 

echo "-----------------------------------------------"
echo "-----------------------------------------------"
python3 main.py $PERM --seed 110 --model eralg4_cv --lr 0.1 --memories 2048 --replay_batch_size 25 
python3 main.py $PERM --seed 110 --model eralg4 --lr 0.1 --memories 2048 --replay_batch_size 25 

echo "-----------------------------------------------"
echo "-----------------------------------------------"
python3 main.py $PERM --seed 115 --model eralg4_cv --lr 0.1 --memories 2048 --replay_batch_size 25 
python3 main.py $PERM --seed 115 --model eralg4 --lr 0.1 --memories 2048 --replay_batch_size 25 

echo "-----------------------------------------------"
echo "-----------------------------------------------"
python3 main.py $PERM --seed 120 --model eralg4_cv --lr 0.1 --memories 2048 --replay_batch_size 25 
python3 main.py $PERM --seed 120 --model eralg4 --lr 0.1 --memories 2048 --replay_batch_size 25 

echo "-----------------------------------------------"
echo "-----------------------------------------------"
python3 main.py $PERM --seed 125 --model eralg4_cv --lr 0.1 --memories 2048 --replay_batch_size 25 
python3 main.py $PERM --seed 125 --model eralg4 --lr 0.1 --memories 2048 --replay_batch_size 25 

echo "-----------------------------------------------"
echo "-----------------------------------------------"
python3 main.py $PERM --seed 130 --model eralg4_cv --lr 0.1 --memories 2048 --replay_batch_size 25 
python3 main.py $PERM --seed 130 --model eralg4 --lr 0.1 --memories 2048 --replay_batch_size 25 

echo "-----------------------------------------------"
echo "-----------------------------------------------"
python3 main.py $PERM --seed 135 --model eralg4_cv --lr 0.1 --memories 2048 --replay_batch_size 25 
python3 main.py $PERM --seed 135 --model eralg4 --lr 0.1 --memories 2048 --replay_batch_size 25 

echo "-----------------------------------------------"
echo "-----------------------------------------------"
python3 main.py $perm --seed 140 --model eralg4_cv --lr 0.1 --memories 2048 --replay_batch_size 25 
python3 main.py $perm --seed 140 --model eralg4 --lr 0.1 --memories 2048 --replay_batch_size 25 

echo "-----------------------------------------------"
echo "-----------------------------------------------"
python3 main.py $perm --seed 145 --model eralg4_cv --lr 0.1 --memories 2048 --replay_batch_size 25 
python3 main.py $perm --seed 145 --model eralg4 --lr 0.1 --memories 2048 --replay_batch_size 25 

echo "-----------------------------------------------"
echo "-----------------------------------------------"
python3 main.py $perm --seed 150 --model eralg4_cv --lr 0.1 --memories 2048 --replay_batch_size 25 
python3 main.py $PERM --seed 150 --model eralg4 --lr 0.1 --memories 2048 --replay_batch_size 25 
