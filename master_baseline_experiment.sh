
declare -A assArray1
assArray1[CE]="1.0"
assArray1[ALPHA]="1.5"
assArray1[FOCAL]="5.0"

mkdir baseline-imbalance-experiment
cd baseline-imbalance-experiment

for loss in "${!assArray1[@]}"; do 
    echo $loss
    mkdir baseline-experiment-$loss
    cp ./../baseline_resnet18_imbalance_experiment.sh baseline-experiment-$loss
    cp ./../losses.py baseline-experiment-$loss
    cp ./../dataset.py baseline-experiment-$loss
    cp ./../baseline-resnet18-imbalance.py baseline-experiment-$loss
    cd baseline-experiment-$loss
    for param in ${assArray1[$loss]}; do
        echo $param
        sbatch --output="baseline_$loss_$param.out" baseline_resnet18_imbalance_experiment.sh $loss $param 
    done
    cd ..
done