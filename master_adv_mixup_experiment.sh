declare -A assArray1
assArray1[CE]="1.0"
assArray1[ALPHA]="0.8 0.9 1.1 1.2 1.3 1.4 1.5 2.0 3.0 4.0"
assArray1[FOCAL]="0.0 0.5 1.0 2.0 5.0"

mkdir adv-mixup-imbalance-experiment
cd adv-mixup-imbalance-experiment

for loss in "${!assArray1[@]}"; do 
    echo $loss
    mkdir adv-mixup-imbalance-experiment-$loss
    cp ./../adv_mixup_experiment.sh adv-mixup-imbalance-experiment-$loss
    cp ./../losses.py adv-mixup-imbalance-experiment-$loss
    cp ./../dataset.py adv-mixup-imbalance-experiment-$loss
    cp ./../av_mixup.py adv-mixup-imbalance-experiment-$loss
    cd adv-mixup-imbalance-experiment-$loss
    for param in ${assArray1[$loss]}; do
        echo $param
        sbatch --output="adv_mixup_${loss}_${param}.out" adv_mixup_experiment.sh $loss $param
    done
    cd ..
done