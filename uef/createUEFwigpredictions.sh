#! /usr/bin/env bash

for i in {5,10,25,50,100,250,500,1000} 
do
	printf "\n Creating predictions with $i documents \n"
	python3.6 ~/repos/IRQPP/uef/uef.py ~/baseline/QL.res lists/list-$i ~/tmp-testing/wig/predictions/predictions-$i -d $i	> wig/predictions-$i
done

