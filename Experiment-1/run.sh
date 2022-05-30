rm results.csv

for m in {0..4}
do
    for s in {0..4}
    do
        python3 get_results.py $s $m
    done
done