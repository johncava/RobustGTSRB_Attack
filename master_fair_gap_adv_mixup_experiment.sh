declare -A assArray1
assArray1[CE]="1.0"
assArray1[ALPHA]="0.95"
assArray1[FOCAL]="5.0"

mkdir fair-gap-adv-mixup-imbalance-experiment
cd fair-gap-adv-mixup-imbalance-experiment

for loss in "${!assArray1[@]}"; do 
    echo $loss
    mkdir fair-gap-adv-mixup-imbalance-experiment-$loss
    cp ./../fair_gap_adv_mixup_experiment.sh fair-gap-adv-mixup-imbalance-experiment-$loss
    cp ./../losses.py fair-gap-adv-mixup-imbalance-experiment-$loss
    cp ./../dataset.py fair-gap-adv-mixup-imbalance-experiment-$loss
    cp ./../fair_gap_av_mixup.py fair-gap-adv-mixup-imbalance-experiment-$loss
    cd fair-gap-adv-mixup-imbalance-experiment-$loss
    for param in ${assArray1[$loss]}; do
        echo $param
        sbatch fair_gap_adv_mixup_experiment.sh $loss $param
    done
    cd ..
done