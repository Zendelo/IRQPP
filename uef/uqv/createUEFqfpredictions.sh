#! /usr/bin/env bash

for i in {5,10,25,50,100,250,500,1000} 
do
	for j in {5,10,25,50,100}
	do
		printf "\n Creating predictions with $i documents \n"
		python3.6 ~/repos/IRQPP/uef/uef.py ~/baseline/QL.res lists/list-$i ~/tmp-testing/qf/predictions/predictions-$i+$j -d $i	> qf/predictions-$i+$j
	done
done

