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