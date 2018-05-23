#! /usr/bin/env bash

#mkdir -v tmp-testing
#printf "The temporary files will be saved in the directory tmp-testing"
for i in {5,10,25,50,100,250,500,1000}
do
	for j in {5,10,25,50,100}
	do
		printf "\n ******** Running for: list $i top $j documents ******** \n"
		./qf.py baseline/QL.res tmp-testing/qf/lists/list-$i -d $j > tmp-testing/qf/predictions/predictions-$i+$j
	done

done

