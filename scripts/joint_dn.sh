datasets="sketch clipart infograph painting quickdraw real"
  

for dataset in $datasets; do
    for learn in .005; do
        for lam in .5 1; do 
                python fine_tune_freeze_dn.py --learning_rate $learn --eval_epochs 5 --experiment_name "$dataset-$learn-$lam" --multi_gpu --dataset $dataset --batch-size 32 --result_path ../results/DomainNet_r34_joint --epochs 30 --lam $lam --joint 
        done                                                                                                                          
    done
done