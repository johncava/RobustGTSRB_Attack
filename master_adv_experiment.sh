
declare -A assArray1
assArray1[CE]="1.0"
assArray1[ALPHA]="1.5"
assArray1[FOCAL]="5.0"

mkdir adv-imbalance-experiment
cd adv-imbalance-experiment

for loss in "${!assArray1[@]}"; do 
    echo $loss
    mkdir adv-experiment-$loss
    cp ./../adv_experiment.sh adv-experiment-$loss
    cp ./../losses.py adv-experiment-$loss
    cp ./../dataset.py adv-experiment-$loss
    cp ./../main.py adv-experiment-$loss
    cd adv-experiment-$loss
    for param in ${assArray1[$loss]}; do
        echo $param
        sbatch adv_experiment.sh $loss $param
    done
    cd ..
done