#! /usr/bin/env bash

#mkdir -v tmp-testing
#printf "The temporary files will be saved in the directory tmp-testing"
for i in {5,10,25,50,100,250,500,1000} 
do
	printf "\n Creating predictions with $i documents \n"
	python3.6 ~/repos/IRQPP/uef/uqv/uef.py ~/baseline/UQVQL.res ~/predictionsUQV/uef/lists/list-$i ~/predictionsUQV/clarity/predictions/predictions-$i -d $i > ~/predictionsUQV/uef/clarity/predictions-$i
done

