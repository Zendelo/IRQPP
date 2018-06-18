#! /usr/bin/env bash
#This script is running the python code that calculates the aggregations

for p in {'clarity','wig','nqc'}
do
	for i in {5,10,25,50,100,250,500,1000}
do
	for j in {'min','max','sum','avg'}
	do
		printf "\n ******** Running for: $i docs $j function ******** \n"
		python3.6 ~/repos/IRQPP/aggregateUQV.py -p ~/predictionsUQV/$p/predictions/predictions-$i -f $j >	~/predictionsUQV/$p/aggr/$j/predictions-$i
	done
done
done

