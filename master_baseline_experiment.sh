
declare -A assArray1
assArray1[CE]="1.0"
assArray1[ALPHA]="0.8 0.9 1.1 1.2 1.3 1.4 1.5 2.0 3.0 4.0"
assArray1[FOCAL]="0.0 0.5 1.0 2.0 5.0"

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
        sbatch --output="baseline_${loss}_${param}.out"  baseline_resnet18_imbalance_experiment.sh $loss $param 
    done
    cd ..
done