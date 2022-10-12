datasets="stanford_cars_cropped cubs_cropped"
  

for dataset in $datasets; do
    for learn in .005 .001 .0005; do
        for lam in .1 .3 ; do 
                python ./train/fine_tune_freeze_pb.py --learning_rate $learn --eval_epochs 5 --experiment_name "$dataset-$learn-$lam" --multi_gpu --dataset $dataset --batch-size 32 --result_path ../results/pb_compare_2 --epochs 30 --lam $lam
        done                                                                                                                          
    done
done