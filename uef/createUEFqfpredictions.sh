#! /usr/bin/env bash

#mkdir -v tmp-testing
#printf "The temporary files will be saved in the directory tmp-testing"
for i in {5,10,25,50,100,250,500,1000} 
do
	for j in {5,10,25,50,100}
	do
		printf "\n Creating predictions with $i documents \n"
		~/repos/IRQPP/uef/uef.py ~/baseline/QL.res ~/tmp-testing/uefnofb/lists/list-$i ~/tmp-testing/qf/predictions/predictions-$i+$j -d $i	> ~/tmp-testing/uef/qf/predictions-$i+$j
	done
done

