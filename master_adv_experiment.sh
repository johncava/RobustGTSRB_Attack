
declare -A assArray1
assArray1[CE]="1.0"
assArray1[ALPHA]="0.8 0.9 1.1 1.2 1.3 1.4 1.5 2.0 3.0 4.0"
assArray1[FOCAL]="0.0 0.5 1.0 2.0 5.0"

Attacks="Square Pixle"

mkdir adv-imbalance-experiment
cd adv-imbalance-experiment

for attack in $Attacks; do
    mkdir $attack
    echo $attack
    cd $attack
    for loss in "${!assArray1[@]}"; do 
        echo $loss
        mkdir adv-experiment-$loss
        cp ./../../adv_experiment.sh adv-experiment-$loss
        cp ./../../losses.py adv-experiment-$loss
        cp ./../../dataset.py adv-experiment-$loss
        cp ./../../main.py adv-experiment-$loss
        cd adv-experiment-$loss
        for param in ${assArray1[$loss]}; do
            echo $param
            sbatch --output="adv_${loss}_${param}.out" adv_experiment.sh $loss $param
        done
        cd ..
    done
    cd ..
done