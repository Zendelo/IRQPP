#! /usr/bin/env bash

for i in {5,10,25,50,100,250,500,1000}
do
	for k in {5,10,25,50,100}
	do
		for j in {'min','max','std','avg'}
			do
		printf "\n ******** Running for: $i docs $j function ******** \n"
		python3.6 ~/repos/IRQPP/aggregateUQV.py -p ./predictions/predictions-$i+$k -f $j >	./aggr/$j/predictions-$i+$k
	done
done
done

