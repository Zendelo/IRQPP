#! /usr/bin/env bash

#mkdir -v tmp-testing
#printf "The temporary files will be saved in the directory tmp-testing"
for i in {5,10,25,50,100,250,500,1000}
do
	printf "\n ******** Running for: $i documents ******** \n"
	./addFBdocs.py -r baseline/QL.res -q data/ROBUST/queries.xml -d $i > uef/tempqueriesUEF-$i
	./addWorkingsetdocs.py baseline/QL.res uef/tempqueriesUEF-$i -d $i > uef/queriesUEF-$i
	printf "\n Running Retrieval \n"
	SetupFiles-indri-5.6/runqueryql/IndriRunQueryQL baseline/indriRunQF.xml -fbDocs=$i uef/queriesUEF-$i > tmp-testing/uef/lists/list-$i
	printf "\n Removing temp queries file \n"
	rm uef/tempqueriesUEF-$i
done


#SetupFiles-indri-5.6/clarity.m-1/Clarity-Fiana clarity/clarityParam.xml -documents=2 data/ROBUST/queries.xml
