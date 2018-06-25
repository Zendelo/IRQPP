#! /usr/bin/env bash
for k in {'MAX','MIN','MEDL','MEDH'}
do
	mkdir -v ~/predictionsUQV/singleUQV/qf$k/lists
	for i in {5,10,25,50,100,250,500,1000}
	do
		printf "\n ******** Running for: $i documents ******** \n"
		~/SetupFiles-indri-5.6/runqueryql/IndriRunQueryQL ~/baseline/indriRunQF.xml -fbDocs=$i ~/data/ROBUST/singleUQV/queries$k.xml > ~/predictionsUQV/singleUQV/qf$k/lists/list-$i
	done
done
#printf "\n Removing filesv \n"



#SetupFiles-indri-5.6/clarity.m-1/Clarity-Fiana clarity/clarityParam.xml -documents=2 data/ROBUST/queries.xml
