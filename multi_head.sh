
for learn in  .01 .005; do
    python fine_tune_multi_dn.py --learning_rate $learn --eval_epochs 5 --experiment_name "$learn" --multi_gpu --batch-size 12 --result_path ../results/joint_DN --epochs 60                                                                                                                
done