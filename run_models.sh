for K in {1..8}
do
    for seed in {1..6}
    do
	for fold in {0..5}
	do
	    screen -S screen_$K$fold$seed "python3 run_model.py $K $fold $seed"
	done
    done
done
