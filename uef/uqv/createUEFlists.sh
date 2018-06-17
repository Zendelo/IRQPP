#! /usr/bin/env bash

for i in {5,10,25,50,100,250,500,1000}
do
	printf "\n ******** Running for: $i documents ******** \n"
	python3.6 ~/repos/IRQPP/addWorkingsetdocs.py ~/baseline/UQVQL.res ~/data/ROBUST/fullqueriesUQV.xml -d $i > ~/uefdataUQV/queriesUQVUEF-$i
	printf "\n Running Retrieval \n"
	~/SetupFiles-indri-5.6/runqueryql/IndriRunQueryQL ~/baseline/indriRunQF.xml -fbDocs=$i ~/uefdataUQV/queriesUQVUEF-$i > ~/predictionsUQV/uef/lists/list-$i
done

