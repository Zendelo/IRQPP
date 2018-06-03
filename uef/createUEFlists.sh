#! /usr/bin/env bash

for i in {5,10,25,50,100,250,500,1000}
do
	printf "\n ******** Running for: $i documents ******** \n"
	./addWorkingsetdocs.py baseline/QL.res data/ROBUST/queries.xml -d $i > uefdata/queriesUEF-$i
	printf "\n Running Retrieval \n"
	SetupFiles-indri-5.6/runqueryql/IndriRunQueryQL baseline/indriRunQF.xml -fbDocs=$i uefdata/queriesUEF-$i > tmp-testing/uef/lists/list-$i
done

