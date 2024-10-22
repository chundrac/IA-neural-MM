for K in {1..8}
do
    for seed in {1..6}
    do
	for fold in {0..5}
	do
	    python3 held_out_loglik.py $K $seed $fold
	done
    done
done
