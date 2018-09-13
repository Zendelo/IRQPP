# IRQPP


The main script is `generate_results.py` \
Usage: `python3.6 generate_results.py -h` \
Before running the code you must ensure having the following files:

_Results Files_
1. ~/QppUqvProj/Results/{corpus}/test/{basic/raw/fused}/QL.res
2. ~/QppUqvProj/Results/{corpus}/test/{basic/raw/fused}/logqlc.res

_Parameters Files_
1. ~/QppUqvProj/Results/{corpus}/test/indriRunQF.xml
2. ~/QppUqvProj/Results/{corpus}/test/indriRunQL.xml
3. ~/QppUqvProj/Results/{corpus}/test/clarityParam.xml

_AP Results Files_
1. ~/QppUqvProj/Results/{corpus}/test/aggregated/map1000-{agg function}
2. ~/QppUqvProj/Results/{corpus}/test/single/map1000-{single selection function}

_Cross Validation files_
1. ~/QppUqvProj/Results/{corpus}/test/2_folds_30_repetitions.json

LogQLC stands for log QL of the query with the entire corpus 

In general the code assumes the directories structure is as seen in the file 
`FS-structure.pdf`

To create the files using indri:\
Create QL.res example for ROBUST UQV will create QL scores retrieved results list: \
`indri-5.6/runqueryql/IndriRunQueryQL QppUqvProj/Results/ROBUST/test/indriRunQL.xml -threads=8 QppUqvProj/data/ROBUST/fullqueriesUQV.xml > QppUqvProj/Results/ROBUST/test/raw/QL.res`\
Create logqlc.res example:\
`indri-5.6/logqlc/LogQlC QppUqvProj/Results/ROBUST/test/indriRunQL.xml QppUqvProj/data/ROBUST/fullqueriesUQV.xml > QppUqvProj/Results/ROBUST/test/raw/logqlc.res`\
Create QLmap1000 (raw ap scores) example:\
`trec_eval -qn -m map QppUqvProj/data/ROBUST/qrelsUQV QppUqvProj/Results/ROBUST/test/raw/QL.res | awk '{print $2, $3}' > QppUqvProj/Results/ROBUST/test/raw/QLmap1000`
Create map1000-max file example (for single pick):
`python3.6 repos/IRQPP/singleUQV.py QppUqvProj/Results/ROBUST/test/raw/QLmap1000 QppUqvProj/Results/ROBUST/test/raw/QLmap1000 -f max > QppUqvProj/Results/ROBUST/test/single/map1000-max`
Create fused (CombSum) results file:\
`python3.7 repos/IRQPP/fusion.py QppUqvProj/Results/ROBUST/test/raw/QL.res QppUqvProj/Results/ROBUST/test/raw/logqlc.res >  QppUqvProj/Results/ROBUST/test/fusion/QL.res`


**Help Files**

Some of the help instructions still need to be updated.


```
usage: python3.6 generate_results.py --predictor PREDICTOR -c CORPUS -q QUERIES 

Full Results Pipeline Automation Generator

optional arguments:
  -h, --help            show this help message and exit
  --predictor predictor_name
                        predictor to run
  -q queries.xml, --queries queries.xml
                        path to queries xml res
  -c {ROBUST,ClueWeb12B}, --corpus {ROBUST,ClueWeb12B}
                        corpus (index) to work with
  --qtype {basic,single,aggregated,fusion}
                        The type of queries to run
  -m {pearson,spearman,kendall}, --measure {pearson,spearman,kendall}
                        default correlation measure type is pearson
  -t {basic,single,aggregated,fusion,all}, --table {basic,single,aggregated,fusion,all}
                        the LaTeX table to be printed
  --generate            generate new predictions
  --lists               generate new lists
  --calc                calc new UQV predictions

Currently Beta Version

```