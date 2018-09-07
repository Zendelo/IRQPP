#! /usr/bin/env bash

for k in {'MAX','MIN','MEDL','MEDH'}
do
	mkdir ~/predictionsUQV/singleUQV/nqc$k/predictions/
	mkdir ~/predictionsUQV/singleUQV/wig$k/predictions/

	for i in {5,10,25,50,100,250,500,1000}
	do
		printf "\n Creating nqc predictions with $i documents \n"
		python3.6 ~/repos/IRQPP/nqc.py ~/baseline/singleUQV/CE$k.res ~/data/ROBUST/singleUQV/queries$k.xml ~/baseline/singleUQV/logqlc$k.res -d $i > ~/predictionsUQV/singleUQV/nqc$k/predictions/predictions-$i

		printf "\n Creating wig predictions with $i documents \n"
		python3.6 ~/repos/IRQPP/wig.py ~/baseline/singleUQV/CE$k.res ~/data/ROBUST/singleUQV/queries$k.xml ~/baseline/singleUQV/logqlc$k.res -d $i > ~/predictionsUQV/singleUQV/wig$k/predictions/predictions-$i
	done
done

