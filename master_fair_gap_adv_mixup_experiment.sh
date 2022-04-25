declare -A assArray1
assArray1[CE]="1.0"
assArray1[ALPHA]="0.8 0.9 1.1 1.2 1.3 1.4 1.5 2.0 3.0 4.0"
assArray1[FOCAL]="0.0 0.5 1.0 2.0 5.0"

Attacks="Square"

mkdir fair-gap-adv-mixup-imbalance-experiment
cd fair-gap-adv-mixup-imbalance-experiment

for attack in $Attacks; do
    mkdir $attack
    echo $attack
    cd $attack
    for loss in "${!assArray1[@]}"; do 
        echo $loss
        mkdir fair-gap-adv-mixup-imbalance-experiment-$loss
        cp ./../../fair_gap_adv_mixup_experiment.sh fair-gap-adv-mixup-imbalance-experiment-$loss
        cp ./../../losses.py fair-gap-adv-mixup-imbalance-experiment-$loss
        cp ./../../dataset.py fair-gap-adv-mixup-imbalance-experiment-$loss
        cp ./../../fair_gap_av_mixup.py fair-gap-adv-mixup-imbalance-experiment-$loss
        cd fair-gap-adv-mixup-imbalance-experiment-$loss
        for param in ${assArray1[$loss]}; do
            echo $param
            sbatch --output="fair_gap_adv_mixup_${loss}_${param}.out" fair_gap_adv_mixup_experiment.sh $loss $param $attack
        done
        cd ..
    done
    cd ..
done