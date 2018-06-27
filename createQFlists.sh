#! /usr/bin/env bash

#mkdir -v tmp-testing
#printf "The temporary files will be saved in the directory tmp-testing"
for i in {5,10,25,50,100,250,500,1000}
do
	printf "\n ******** Running for: $i documents ******** \n"
	~/SetupFiles-indri-5.6/runqueryql/IndriRunQueryQL ~/baseline/indriRunQF.xml -fbDocs=$i data/ROBUST/queries.xml > basicPredictions/qf/lists/list-$i
done
#printf "\n Removing filesv \n"


#~/SetupFiles-indri-5.6/runqueryql/IndriRunQueryQL
#SetupFiles-indri-5.6/clarity.m-1/Clarity-Fiana clarity/clarityParam.xml -documents=2 data/ROBUST/queries.xml
