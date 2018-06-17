#! /usr/bin/env bash

for i in {5,10,25,50,100,250,500,1000} 
do
	printf "\n Creating predictions with $i documents \n"
	python3.6 ~/repos/IRQPP/uef/uqv/uef.py ~/baseline/UQVQL.res ~/predictionsUQV/uef/lists/list-$i ~/predictionsUQV/wig/predictions/predictions-$i -d $i > ~/predictionsUQV/uef/wig/predictions/predictions-$i
done

