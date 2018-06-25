#! /usr/bin/env bash

for k in {'MAX','MIN','MEDL','MEDH'}
do
	mkdir -v ~/predictionsUQV/singleUQV/qf$k/predictions
	for i in {5,10,25,50,100,250,500,1000}
	do
		for j in {5,10,25,50,100}
		do
			printf "\n ******** Running for: list $i top $j documents ******** \n"
			python3.6 ~/repos/IRQPP/qf.py ~/baseline/singleUQV/QL$k.res ~/predictionsUQV/singleUQV/qf$k/lists/list-$i -d $j > ~/predictionsUQV/singleUQV/qf$k/predictions/predictions-$i+$j
		done
		
	done

done

