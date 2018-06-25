#! /usr/bin/env bash

for k in {'MAX','MIN','MEDL','MEDH'}
do
	mkdir -v ~/predictionsUQV/singleUQV/clarity$k/predictions
	for i in {5,10,25,50,100,250,500,1000}
	do
			printf "\n ******** Running for: list $i top $j documents ******** \n"
			~/SetupFiles-indri-5.6/clarity.m-2/Clarity-Anna ~/clarity/clarityParam.xml -fbDocs=$i ~/data/ROBUST/singleUQV/queries$k.xml > ~/predictionsUQV/singleUQV/clarity$k/predictions/predictions-$i
		
	done

done


