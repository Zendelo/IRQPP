#!/usr/bin/env bash
for i in {5,10,25,50,100,250,500,1000}
do printf "\n ******** Running for: $i documents ******** \n"
SetupFiles-indri-5.6/clarity.m-1/Clarity-Fiana clarity/clarityParam.xml -documents=$i data/ROBUST/queries.xml > baseline/testing-$i
correlation.py -f baseline/QLmap1000 -s baseline/testing-$i -t pearson
done

#SetupFiles-indri-5.6/clarity.m-1/Clarity-Fiana clarity/clarityParam.xml -documents=2 data/ROBUST/queries.xml