#!/usr/bin/env bash
mkdir -v tmp-testing
printf "The temporary files will be saved in the directory tmp-testing"
for i in {5,10,25,50,100,250,500,1000}
do
	printf "\n ******** Running for: $i documents ******** \n"
	SetupFiles-indri-5.6/clarity.m-1/Clarity-Fiana clarity/clarityParam.xml -documents=$i data/ROBUST/queries.xml > tmp-testing/testing-$i
	python correlation.py baseline/QLmap1000  tmp-testing/testing-$i -t pearson
	python correlation.py baseline/QLmap1000  tmp-testing/testing-$i -t spearman
	python correlation.py baseline/QLmap1000  tmp-testing/testing-$i -t kendall
done
printf "\n Removing filesv \n"

rm -rfv tmp-testing

#SetupFiles-indri-5.6/clarity.m-1/Clarity-Fiana clarity/clarityParam.xml -documents=2 data/ROBUST/queries.xml
