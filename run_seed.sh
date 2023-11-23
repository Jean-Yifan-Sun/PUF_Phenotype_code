source /home/sunyf23/anaconda3/etc/profile.d/conda.sh
conda init bash
conda activate mygcn

for((i=2;i<=5;i=i+1));  
do
let seed=$i*123454321
 /home/sunyf23/anaconda3/envs/mygcn/bin/python /home/sunyf23/Work_station/PUF_Phenotype/runner.py --optim_type sgd --seed $seed

 /home/sunyf23/anaconda3/envs/mygcn/bin/python /home/sunyf23/Work_station/PUF_Phenotype/random_inference.py --seed $seed

 done