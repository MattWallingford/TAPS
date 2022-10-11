datasets="sketches stanford_cars_cropped flowers cubs_cropped wikiart"
  

for dataset in $datasets; do
    for learn in .00005; do
        for lam in 1; do 
                python fine_tune_freeze_pb.py --learning_rate $learn --lam $lam --eval_epochs 5 --experiment_name "$dataset-$learn-$lam" --multi_gpu --dataset $dataset --batch-size 32 --result_path ../results/transformer --epochs 30 --Vit True --warmup_epochs 2
        done                                                                                                                          
    done
done