#! /usr/bin/env bash

#mkdir -v tmp-testing
#printf "The temporary files will be saved in the directory tmp-testing"
for i in {5,10,25,50,100,250,500,1000} 
do
	printf "Creating predictions with $i documents"
	~/uef.py ~/baseline/QL.res ../lists/list-$i ../../clarity-Anna/predictions/predictions-$i -d $i	> predictions-$i
done

