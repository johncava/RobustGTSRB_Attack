declare -A assArray1
#assArray1[CE]="1.0"
assArray1[ALPHA]="0.95"
#assArray1[FOCAL]="5.0"

# mkdir adv-mixup-imbalance-experiment
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
        sbatch adv_mixup_experiment.sh $loss $param
    done
    cd ..
done