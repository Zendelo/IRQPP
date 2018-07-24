# IRQPP

The main 'pipe' script is `generate_results.py`
Before running the code you must ensure having the following files:

_Results Files_
1. ~/QppUqvProj/Results/{corpus}/test/{queries type}/QL.res
2. ~/QppUqvProj/Results/{corpus}/test/{queries type}/logqlc.res

_Parameters Files_
1. ~/QppUqvProj/Results/{corpus}/test/indriRunQF.xml
2. ~/QppUqvProj/Results/{corpus}/test/indriRunQL.xml
3. ~/QppUqvProj/Results/{corpus}/test/clarityParam.xml

LogQLC stands for QL of the query with the entire corpus



The evaluation code is crosseval.py
The code to add FB documents to a given query xml file is addFBdocs.py


**Help Files**

Some of the help instructions still needs to be updated.


```
./crossval.py -h
usage: Use CV to optimize correlation

Cross Validation script

optional arguments:
  -h, --help            show this help message and exit
  -p predictions_dir, --predictions predictions_dir
                        path to prediction results files directory
  --labeled LABELED     path to labeled list res
  -r REPEATS, --repeats REPEATS
                        number of repeats
  -k SPLITS, --splits SPLITS
                        number of k-fold
  -m {pearson,spearman,kendall}, --measure {pearson,spearman,kendall}
                        default correlation measure type is pearson
  -g, --generate        generate new CrossValidation sets
  -l CV_FILE_PATH, --load CV_FILE_PATH
                        load existing CrossValidation JSON res

Prints the average correlation
```
```
./wig.py -h
usage: Input CE(q|d) scores and queries files

WIG predictor

positional arguments:
  CE(q|d)_results_file  The CE results file for the documents scores
  queries_xml_file      The queries xml file
  QLC                   The queries xml file

optional arguments:
  -h, --help            show this help message and exit
  -t queries_txt_file, --textqueries queries_txt_file
                        The queries txt file
  -d fbDocs, --docs fbDocs
                        Number of documents

Prints the WIG predictor scores
```
```
./nqc.py -h
usage: Input CE(q|d) scores and queries files

NQC predictor

positional arguments:
  CE(q|d)_results_file  The CE results file for the documents scores
  queries_xml_file      The queries xml file
  QLC                   The queries xml file

optional arguments:
  -h, --help            show this help message and exit
  -t queries_txt_file, --textqueries queries_txt_file
                        The queries txt file
  -d fbDocs, --docs fbDocs
                        Number of documents

Prints the NQC predictor scores
```
```
./addFBdocs.py -h
usage: Use CV to optimize correlation

Cross Validation script

optional arguments:
  -h, --help            show this help message and exit
  -r results_file, --results results_file
                        The results file for the Relevance Feedback
  -q queries_xml_file, --queries queries_xml_file
                        The queries xml file
  -d fbDocs, --docs fbDocs
                        Number of Feedback documents to add

Prints the average correlation
```