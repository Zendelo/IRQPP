/*==========================================================================
* Copyright (c) 2004 University of Massachusetts.  All Rights Reserved.
*
* Use of the Lemur Toolkit for Language Modeling and Information Retrieval
* is subject to the terms of the software license set forth in the LICENSE
* file included with this software, and also available at
* http://www.lemurproject.org/license.html
*
*==========================================================================
*/


#include <time.h>
#include "indri/QueryEnvironment.hpp"
#include "indri/LocalQueryServer.hpp"
#include "indri/delete_range.hpp"
#include "indri/NetworkStream.hpp"
#include "indri/NetworkMessageStream.hpp"
#include "indri/NetworkServerProxy.hpp"

#include "indri/ListIteratorNode.hpp"
#include "indri/ExtentInsideNode.hpp"
#include "indri/DocListIteratorNode.hpp"
#include "indri/FieldIteratorNode.hpp"

#include "indri/Parameters.hpp"

#include "indri/ParsedDocument.hpp"
#include "indri/Collection.hpp"
#include "indri/CompressedCollection.hpp"
#include "indri/TaggedDocumentIterator.hpp"
#include "indri/XMLNode.hpp"

#include "indri/QueryExpander.hpp"
#include "indri/RMExpander.hpp"
#include "indri/PonteExpander.hpp"
// need a QueryExpanderFactory....
#include "indri/TFIDFExpander.hpp"

#include "indri/IndriTimer.hpp"
#include "indri/UtilityThread.hpp"
#include "indri/ScopedLock.hpp"
#include "indri/delete_range.hpp"
#include "indri/SnippetBuilder.hpp"

#include "indri/RelevanceModel.hpp"
#include <cmath>

#include <queue>
#include <set>

static bool copy_parameters_to_string_vector( std::vector<std::string>& vec, indri::api::Parameters p, const std::string& parameterName ) {
  if( !p.exists(parameterName) )
  return false;

  indri::api::Parameters slice = p[parameterName];

  for( size_t i=0; i<slice.size(); i++ ) {
    vec.push_back( slice[i] );
  }

  return true;
}

struct query_t {
  struct greater {
    bool operator() ( query_t* one, query_t* two ) {
      return one->index > two->index;
    }
  };

  query_t( int _index, std::string _number, const std::string& _text, const std::string &queryType,  std::vector<std::string> workSet,   std::vector<std::string> FBDocs) :
  index( _index ),
  number( _number ),
  text( _text ), qType(queryType), workingSet(workSet), relFBDocs(FBDocs)
  {
  }

  query_t( int _index, std::string _number, const std::string& _text ) :
  index( _index ),
  number( _number ),
  text( _text )
  {
  }

  std::string number;
  int index;
  std::string text;
  std::string qType;
  // working set to restrict retrieval
  std::vector<std::string> workingSet;
  // Rel fb docs
  std::vector<std::string> relFBDocs;
};

class QueryThread : public indri::thread::UtilityThread {
private:
  indri::thread::Lockable& _queueLock;
  indri::thread::ConditionVariable& _queueEvent;
  std::queue< query_t* >& _queries;
  std::priority_queue< query_t*, std::vector< query_t* >, query_t::greater >& _output;

  indri::api::QueryEnvironment _environment;
  indri::api::Parameters& _parameters;
  int _requested;
  int _initialRequested;

  bool _printDocuments;
  bool _printPassages;
  bool _printSnippets;
  bool _printQuery;

  std::string _runID;
  bool _trecFormat;
  bool _inexFormat;

  indri::query::QueryExpander* _expander;
  std::vector<indri::api::ScoredExtentResult> _results;
  indri::api::QueryAnnotation* _annotation;


  void parse_query( const std::string& query, std::vector<std::string>& qset )
  {
    qset.clear();
    std::string qcopy( query);
    for ( size_t i=0; i<qcopy.size(); i++ )
    {
      if ( (qcopy[i]=='(')||(qcopy[i]==')')||(qcopy[i]=='.')||(qcopy[i]==',')||(qcopy[i]==':'))
      {
        qcopy[i]=' ';
      }
    }
    std::istringstream oss (qcopy);
    //oss << qcopy;
    while (oss.good())
    {
      std::string temp;
      oss >> temp;
      if ( temp.empty()) continue;
      if ( temp.find('#')!= std::string::npos) continue;
      //stem here
      qset.push_back( _environment.stemTerm( temp) );
    }

  }


  //This is the function that calculates Clarity
  //In fact it calculates the following properties of the RM:
  //Entropy, Cross-entropy (CE) and KL to the corpus language model
  //KL(RM||Corpus)= CE(RM|Corpus) - Entropy(RM)
  void _runQuery( std::stringstream& output, const std::string& query,
    const std::string &queryType, const std::string& q_number) {
      try {

        _results = _environment.runQuery( query, _initialRequested, queryType );
        std::vector<std::string> qset;
        parse_query( query, qset );

        //The number of documents in the RM
        int fbDocs = _parameters.get( "fbDocs" , 10 );
        //The term cutoff of the RM
        int fbTerms = _parameters.get( "fbTerms" , 100 );
        //the weight of the original query - usually 0 should be given in Clarity
        double fbOrigWt = _parameters.get( "fbOrigWeight", 0 );
        //Per my previous results it is better not to smooth the RM
        std::string rmSmoothing = _parameters.get("smoothing", "method:jm,lambda:0.0");
        //1- rocchio, default 0 -QL scores
        int iRocchio = _parameters.get( "fbDocumentWeights", 0);
        if (iRocchio > 0) //give all documents the same weight
        {
          for (size_t i=0; i<_results.size(); i++ )
          {
            _results[i].score=1.0;
          }
        }
        else//fix the scores to be QL scores
        {
          for (size_t i=0; i<_results.size(); i++ )
          {
            _results[i].score= _results[i].score*qset.size();
          }
        }
        // Creates an unordered set of query terms
        std::set<std::string> query_terms(qset.begin(),qset.end());
        // std::copy(qset.begin(), qset.end(), std::inserter(query_terms,query_terms.begin()));

        int maxGrams = 1;
        //build a relevance model
        indri::query::RelevanceModel rm( _environment, rmSmoothing, maxGrams, _results.size());
        rm.generate( query, _results );

        //iterate over the model and find query terms
        //	TODO: Consider change the reference to copy by value
        const std::vector<indri::query::RelevanceModel::Gram*>& grams = rm.getGrams();

        //Define clipping parameter
        size_t limit = ( grams.size() < fbTerms ) ? grams.size() : fbTerms;
        double dSumRM(0.f);

        // Sum all the probabilities of the top `fbTerms` words
        for( size_t j=0; j < limit; j++ ) {
          dSumRM += grams[j]->weight;
        }

        double totalTermCount = (double)_environment.termCount();

        double entropy(0.0), cross_entropy(0.0);

        size_t query_length = qset.size();
        int limit_bound = limit;

        // sorted grams came from rm
        for( size_t j=0; j< grams.size(); j++ ) {

          limit_bound--;

          std::string &term = grams[j]->terms[0];
          // check if the term exists in the query
          if (query_terms.find(term) == query_terms.end())
            continue;

          double pwq = grams[j]->weight;
          double pwq_clipped(0.0);

          // check if the term is in the clipped bounds
          if (limit_bound >= 0)
            pwq_clipped = pwq/dSumRM;

          output << q_number << " "  << term << " " << pwq_clipped << " " <<  fbTerms << std::endl;
          output << q_number << " "  << term << " " << pwq << " " <<  "nan" << std::endl;

          query_length-- ;
          // check if all query terms were summed
          if (query_length < 1)
            break;
//          entropy += pwq*(log(pwq)/log(2.0));
//          cross_entropy += pwq*(log(pw)/log(2.0));
        }

//        cross_entropy = -cross_entropy;
//        entropy = -entropy;

//        double dClarity = cross_entropy - entropy;
//        output << q_number << "clip"  << entropy  << " " <<  cross_entropy << " " << dClarity << std::endl;
//        output << q_number << " "  << entropy  << " " <<  cross_entropy << " " << dClarity << std::endl;
        _results.clear();
      }
      catch( lemur::api::Exception& e )
      {
        _results.clear();
        LEMUR_RETHROW(e, "Clarity function crashed with Exception");
      }
    }

  public:
    QueryThread( std::queue< query_t* >& queries,
      std::priority_queue< query_t*, std::vector< query_t* >, query_t::greater >& output,
      indri::thread::Lockable& queueLock,
      indri::thread::ConditionVariable& queueEvent,
      indri::api::Parameters& params ) :
      _queries(queries),
      _output(output),
      _queueLock(queueLock),
      _queueEvent(queueEvent),
      _parameters(params),
      _expander(0),
      _annotation(0)
      {
      }

      ~QueryThread() {
      }

      UINT64 initialize() {
        _environment.setSingleBackgroundModel( _parameters.get("singleBackgroundModel", false) );

        std::vector<std::string> stopwords;
        if( copy_parameters_to_string_vector( stopwords, _parameters, "stopper.word" ) )
        _environment.setStopwords(stopwords);

        std::vector<std::string> smoothingRules;
        if( copy_parameters_to_string_vector( smoothingRules, _parameters, "rule" ) )
        _environment.setScoringRules( smoothingRules );

        if( _parameters.exists( "index" ) ) {
          indri::api::Parameters indexes = _parameters["index"];

          for( size_t i=0; i < indexes.size(); i++ ) {
            _environment.addIndex( std::string(indexes[i]) );
          }
        }

        if( _parameters.exists( "server" ) ) {
          indri::api::Parameters servers = _parameters["server"];

          for( size_t i=0; i < servers.size(); i++ ) {
            _environment.addServer( std::string(servers[i]) );
          }
        }

        if( _parameters.exists("maxWildcardTerms") )
        _environment.setMaxWildcardTerms(_parameters.get("maxWildcardTerms", 100));

        _requested = _parameters.get( "count", 1000 );
        _initialRequested = _parameters.get( "fbDocs", _requested );
        _runID = _parameters.get( "runID", "indri" );
        _trecFormat = _parameters.get( "trecFormat" , false );
        _inexFormat = _parameters.exists( "inex" );

        _printQuery = _parameters.get( "printQuery", false );
        _printDocuments = _parameters.get( "printDocuments", false );
        _printPassages = _parameters.get( "printPassages", false );
        _printSnippets = _parameters.get( "printSnippets", false );

        if (_parameters.exists("baseline")) {
          // doing a baseline
          std::string baseline = _parameters["baseline"];
          _environment.setBaseline(baseline);
          // need a factory for this...
          if( _parameters.get( "fbDocs", 0 ) != 0 ) {
            // have to push the method in...
            std::string rule = "method:" + baseline;
            _parameters.set("rule", rule);
            _expander = new indri::query::TFIDFExpander( &_environment, _parameters );
          }
        } else {
          if( _parameters.get( "fbDocs", 0 ) != 0 ) {
            _expander = new indri::query::RMExpander( &_environment, _parameters );
          }
        }

        if (_parameters.exists("maxWildcardTerms")) {
          _environment.setMaxWildcardTerms((int)_parameters.get("maxWildcardTerms"));
        }
        return 0;
      }

      void deinitialize() {
        delete _expander;
        _environment.close();
      }

      bool hasWork() {
        indri::thread::ScopedLock sl( &_queueLock );
        return _queries.size() > 0;
      }

      UINT64 work() {
        query_t* query;
        std::stringstream output;

        // pop a query off the queue
        {
          indri::thread::ScopedLock sl( &_queueLock );
          if( _queries.size() ) {
            query = _queries.front();
            _queries.pop();
          } else {
            return 0;
          }
        }

        // run the query
        try {
          if (_parameters.exists("baseline") && ((query->text.find("#") != std::string::npos) || (query->text.find(".") != std::string::npos)) ) {
            LEMUR_THROW( LEMUR_PARSE_ERROR, "Can't run baseline on this query: " + query->text + "\nindri query language operators are not allowed." );
          }
          _runQuery( output, query->text, query->qType, query->number);
        } catch( lemur::api::Exception& e ) {
          output << "# EXCEPTION in query " << query->number << ": " << e.what() << std::endl;
        }

        // print the results to the output stream
        //  _printResults( output, query->number );

        // push that data into an output queue...?
        {
          indri::thread::ScopedLock sl( &_queueLock );
          _output.push( new query_t( query->index, query->number, output.str() ) );
          _queueEvent.notifyAll();
        }

        delete query;
        return 0;
      }
    };

    void push_queue( std::queue< query_t* >& q, indri::api::Parameters& queries,
      int queryOffset ) {

        for( size_t i=0; i<queries.size(); i++ ) {
          std::string queryNumber;
          std::string queryText;
          std::string queryType = "indri";
          if( queries[i].exists( "type" ) )
          queryType = (std::string) queries[i]["type"];
          if (queries[i].exists("text"))
          queryText = (std::string) queries[i]["text"];
          if( queries[i].exists( "number" ) ) {
            queryNumber = (std::string) queries[i]["number"];
          } else {
            int thisQuery=queryOffset + int(i);
            std::stringstream s;
            s << thisQuery;
            queryNumber = s.str();
          }
          if (queryText.size() == 0)
          queryText = (std::string) queries[i];

          // working set and RELFB docs go here.
          // working set to restrict retrieval
          std::vector<std::string> workingSet;
          // Rel fb docs
          std::vector<std::string> relFBDocs;
          copy_parameters_to_string_vector( workingSet, queries[i], "workingSetDocno" );
          copy_parameters_to_string_vector( relFBDocs, queries[i], "feedbackDocno" );

          q.push( new query_t( i, queryNumber, queryText, queryType, workingSet, relFBDocs ) );

        }
      }

      int main(int argc, char * argv[]) {
        try {
          indri::api::Parameters& param = indri::api::Parameters::instance();
          param.loadCommandLine( argc, argv );

          if( param.get( "version", 0 ) ) {
            std::cout << INDRI_DISTRIBUTION << std::endl;
          }

          if( !param.exists( "query" ) )
          LEMUR_THROW( LEMUR_MISSING_PARAMETER_ERROR, "Must specify at least one query." );

          if( !param.exists("index") && !param.exists("server") )
          LEMUR_THROW( LEMUR_MISSING_PARAMETER_ERROR, "Must specify a server or index to query against." );

          if (param.exists("baseline") && param.exists("rule"))
          LEMUR_THROW( LEMUR_BAD_PARAMETER_ERROR, "Smoothing rules may not be specified when running a baseline." );

          int threadCount = param.get( "threads", 1 );
          std::queue< query_t* > queries;
          std::priority_queue< query_t*, std::vector< query_t* >, query_t::greater > output;
          std::vector< QueryThread* > threads;
          indri::thread::Mutex queueLock;
          indri::thread::ConditionVariable queueEvent;

          // push all queries onto a queue
          indri::api::Parameters parameterQueries = param[ "query" ];
          int queryOffset = param.get( "queryOffset", 0 );
          push_queue( queries, parameterQueries, queryOffset );
          int queryCount = (int)queries.size();

          // launch threads
          for( int i=0; i<threadCount; i++ ) {
            threads.push_back( new QueryThread( queries, output, queueLock, queueEvent, param ) );
            threads.back()->start();
          }

          int query = 0;

          bool inexFormat = param.exists( "inex" );
          if( inexFormat ) {
            std::string participantID = param.get( "inex.participantID", "1");
            std::string runID = param.get( "runID", "indri" );
            std::string inexTask = param.get( "inex.task", "CO.Thorough" );
            std::string inexTopicPart = param.get( "inex.topicPart", "T" );
            std::string description = param.get( "inex.description", "" );
            std::string queryType = param.get("inex.query", "automatic");
            std::cout << "<inex-submission participant-id=\"" << participantID
            << "\" run-id=\"" << runID
            << "\" task=\"" << inexTask
            << "\" query=\"" << queryType
            << "\" topic-part=\"" << inexTopicPart
            << "\">" << std::endl
            << "  <description>" << std::endl << description
            << std::endl << "  </description>" << std::endl;
          }

          // acquire the lock.
          queueLock.lock();

          // process output as it appears on the queue
          while( query < queryCount ) {
            query_t* result = NULL;

            // wait for something to happen
            queueEvent.wait( queueLock );

            while( output.size() && output.top()->index == query ) {
              result = output.top();
              output.pop();

              queueLock.unlock();

              std::cout << result->text;
              delete result;
              query++;

              queueLock.lock();
            }
          }
          queueLock.unlock();

          if( inexFormat ) {
            std::cout << "</inex-submission>" << std::endl;
          }

          // join all the threads
          for( size_t i=0; i<threads.size(); i++ )
          threads[i]->join();

          // we've seen all the query output now, so we can quit
          indri::utility::delete_vector_contents( threads );
        } catch( lemur::api::Exception& e ) {
          LEMUR_ABORT(e);
        } catch( ... ) {
          std::cout << "Caught unhandled exception" << std::endl;
          return -1;
        }

        return 0;
      }
