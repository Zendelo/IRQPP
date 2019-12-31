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

//
// dumpindex
//
// 13 September 2004 -- tds
//

#include "indri/Repository.hpp"
#include "indri/CompressedCollection.hpp"
#include "indri/LocalQueryServer.hpp"
#include "indri/ScopedLock.hpp"
#include "indri/QueryEnvironment.hpp"
#include <iostream>
#include <math.h>

//////////////////////My code/////////////////////////

typedef vector<double> double_vector;

double Average( const double_vector& xv )
{
	double dSum(0);
	for ( double_vector::const_iterator it = xv.begin();
		  it != xv.end();
		  ++it )
	{
		dSum += (*it);
	}
	return (dSum/xv.size());
}

double Correl( const double_vector& xv, const double_vector& yv )
{
	double dAvgX = Average( xv );
	double dAvgY = Average( yv );
	
	double dUpper(0);
	double dLowerX(0);
	double dLowerY(0);

	double_vector::const_iterator citY= yv.begin();
	for ( double_vector::const_iterator citX = xv.begin() ;
		citX != xv.end(); ++citX )
	{
		dUpper += ( ((*citX) - dAvgX )*((*citY) - dAvgY ) );
		dLowerX += pow( ((*citX) - dAvgX ), 2.0 );
		dLowerY += pow( ((*citY) - dAvgY), 2.0 );
		++citY;
	}

	return ( dUpper/sqrt(dLowerX*dLowerY) );
}


double Sum(const double_vector& xv )
{
	double dSum(0);
	for ( double_vector::const_iterator it = xv.begin();
		  it != xv.end();
		  ++it )
	{
		dSum += (*it);
	}
	return dSum;
}

double Strdev( const double_vector& xv )
{
	if ( xv.size() <= 1 ) return 0;
	double dSum(0);
	double dSumSquare(0);
	for ( double_vector::const_iterator it = xv.begin();
		  it != xv.end();
		  ++it )
	{
		dSum += (*it);
		dSumSquare += (*it)*(*it);
	}
	double dOne = (dSumSquare/(xv.size()));
   	double dTwo = ((dSum*dSum)/((double)xv.size()*(double)xv.size()));
	double dVal=sqrt(dOne- dTwo);
	return dVal;
}

//---------------------------------Kendalls---------------------------
void CountTies(const double_vector& dv, double_vector& ties_count)
{
	if (dv.empty()) return;
	ties_count.clear();
	double_vector dv_copy;
	copy( dv.begin(), dv.end(), back_inserter(dv_copy) );
	sort( dv_copy.begin(), dv_copy.end() );
	double in_count = dv_copy[0];
	double count=1;
	for( size_t i=1; i < dv_copy.size(); i++ )
	{
		//starting a count for a new group
		if ( in_count != dv_copy[i] )
		{
			ties_count.push_back( count );
			count = 1;
			in_count = dv_copy[i];
		}//the group already exists
		else if ( in_count == dv_copy[i] )
		{
			count++;
		}
	}
    ties_count.push_back( count );//last one was not inserted

}
double TiedSumForKendall( const double_vector& ties)
{
	double dsum(0.0);
	for ( size_t i= 0; i< ties.size(); i++ )
	{
		dsum += ((ties[i]*(ties[i]-1.0))/2.0);
	}
	return dsum;
}

double KendallsTau( const double_vector& xv, const double_vector& yv )
{
	double Nc(0.0), Nd(0.0);
	if (xv.size()!=yv.size() )return 0;
	size_t N( xv.size() );

	for ( size_t i=0; i < N ; ++i )
	{
		for ( size_t j=(i+1); j < N ; ++j)
		{
			if (( ((xv[i]-xv[j]) <0 )&& ((yv[i]-yv[j]) <0 )) || 
				  (((xv[i]-xv[j]) >0 )&& ((yv[i]-yv[j]) >0 )))
				  Nc++;
			else if (!(((xv[i]-xv[j]) ==0 )|| ( (yv[i]-yv[j]) ==0 ))) Nd++;
		//	else cout << "tiko" << endl;
		}
	}
	double_vector x_ties, y_ties;
	CountTies( xv, x_ties );
	CountTies( yv, y_ties );
	double n_0 = (double)N*((double)N-1.0)/2.0;
	double denom = sqrt((n_0-TiedSumForKendall(x_ties))*(n_0-TiedSumForKendall(y_ties)));
	return (( Nc - Nd )/(denom));
}
//------------------------------------------------------------------------

bool ReadQuery(  ifstream& in, string& queryid, set<string>& terms, indri::api::QueryEnvironment& env )
{
     string query;
     terms.clear();
      if (!in.good())return false;
     getline( in, query );
     if (query.empty())return false;
     stringstream split;
     split <<  query;
     split >> queryid;
     while (split.good())
     {
           string term;
           split >>  term;
           string::iterator cit = term.begin();
           while ( *cit ==' ') cit++;
           string tcopy;
           std::copy( cit, term.end(), back_inserter( tcopy ) );
           if ( !tcopy.empty() )   terms.insert( env.stemTerm(  tcopy) );
     }
     if ( terms.size()==0) return false;
  //    cerr << "after terms loop in ReadQuery" << endl;
     return true;
}
void WriteQidValueFile( const string& file_name, const vector<string>& query_ids, const double_vector& values )
{
     ofstream out;     
     out.open( file_name.c_str(), ofstream::out );
     for ( size_t i=0; i< query_ids.size(); i++) 
     {
         out << query_ids[i] << " " << values[i] << endl;    
     }
     out.close();
}
void  ReadQidValueFile( const string& file_name, vector<string>& query_ids, double_vector& values )
{
     ifstream in;
     in.open( file_name.c_str(), ifstream::in );
     while (in.good())
     {
         string qid; 
         double dvalue(0.0);
         in >> qid >> dvalue;
         query_ids.push_back( qid );
         values.push_back( dvalue );
     }
     in.close();
}



void MergeQIDs( const vector<string>& left, const vector<string>& right, std::set<size_t>& tofilterLeft, std::set<size_t>& tofilterRight  )
{
     std::set<string> set_left, set_right;
     std::copy( left.begin(), left.end(), inserter( set_left, set_left.end() ));
     std::copy( right.begin(), right.end(), inserter( set_right, set_right.end() ));

     for ( vector<string>::size_type l=0; l<left.size(); l++ )
     {
         if ( set_right.find( left[l])==set_right.end() )
            tofilterLeft.insert( l);
     }  
     
     for ( vector<string>::size_type r=0; r<right.size(); r++ )
     {
         if ( set_left.find( right[r])==set_left.end() )
            tofilterRight.insert( r );
     }  
  
}

void Filter( double_vector& values, const std::set<size_t>& tofilter )
{
     double_vector filtered;
     for ( double_vector::size_type l=0; l<values.size(); l++)
     {
         if ( tofilter.find( l) == tofilter.end() )
         {
              filtered.push_back( values[l]);
               
          }
      }
      values = filtered;
}
//////////////////////////////////////---My code---/////////////////////////////////////////



void GetStatisticsInDocs(const std::string& term, UINT64& docCount, double_vector& dv_tf, double_vector& dv_doc_length, indri::collection::Repository& r )
{
      indri::collection::Repository::index_state state = r.indexes();
      dv_tf.clear();
      dv_doc_length.clear();
      
      for( size_t i=0; i<state->size(); i++ ) {
          
          indri::index::Index* index = (*state)[i];
          indri::thread::ScopedLock( index->iteratorLock() );

          indri::index::DocListIterator* iter = index->docListIterator( term ); //term should be preprocessed
          if (iter == NULL) continue;

          iter->startIteration();

          int doc = 0;
          indri::index::DocListIterator::DocumentData* entry;

          for( iter->startIteration(); iter->finished() == false; iter->nextEntry() )
           {
                  docCount++;
                  entry = iter->currentEntry();
                  dv_tf.push_back( entry->positions.size() );
                  dv_doc_length.push_back( index->documentLength( entry->document ) );
           }

           delete iter;
     }
}



int main( int argc, char** argv ) {
  try {
    if (argc < 4) {
    cerr << "usage: PreRetrievalIndri index_path plain_queries_path eval_measure_path dir_prior" << endl;
    cerr << "plain_queries should be each line new query in format: qid term_1 term_2 .. term_k " << endl;
    cerr << "eval_measure should be each line new query in format: qid value" <<endl;
    exit (1);
     }

    indri::collection::Repository r;
    std::string repName = argv[1];
    r.openRead( repName );

	//
	indri::api::QueryEnvironment env;
	env.addIndex( repName );
    
    ifstream in;
    in.open(argv[2] ,ifstream::in);
    string eval_file( argv[3] );
    int iDirPrior( atoi( argv[4] ) );
  
   string query_id;
   set<string> qterms;
  ///predictor outputs
   vector<string> qids;
   vector<double> sqc_sum_tf, sqc_avg_tf, sqc_max_tf;
   vector<double> sqc_sum_lm, sqc_avg_lm, sqc_max_lm;
   vector<double> var_sum_tf, var_avg_tf, var_max_tf;
   vector<double> var_sum_lm, var_avg_lm, var_max_lm;
   vector<double> sum_idf, avg_idf, max_idf;

   indri::server::LocalQueryServer local(r);
   UINT64 totalTermCount = local.termCount();
   UINT64 docCountTotal = local.documentCount();
  
  while (ReadQuery( in, query_id, qterms, env ))
  {
        qids.push_back( query_id ); 
        
         cerr << "Read query id: " << query_id << endl;
  
        /// Similarity predictors
      
        double_vector vect_idf(qterms.size(), 0.0);
        double_vector vect_tf_idf1(qterms.size(), 0.0);
        double_vector vect_lm1(qterms.size(), 0.0);
        double_vector vect_tf_idf2(qterms.size(), 0.0);
        double_vector vect_lm2(qterms.size(), 0.0);
        int i=0;
        for (set<string>::const_iterator cit = qterms.begin();cit != qterms.end(); cit++ )
        {
            UINT64 termCount = local.termCount( *cit );
            if ( termCount==0) continue;
            UINT64 docCount = 0.0;
            double_vector dv_tf;
            double_vector dv_doc_length;
            GetStatisticsInDocs( *cit, docCount, dv_tf, dv_doc_length, r );          
            //for SQC and IDF
            vect_lm1[i]= log( (double)(termCount)/(double)(totalTermCount));
            vect_tf_idf1[i]= ( 1.0+ log(termCount) )*( log( 1.0 + (double)docCountTotal/(double)docCount ));
            vect_idf[i] = log( 1.0 + docCountTotal/(double)docCount );
            
            double_vector doc_tf_idf, doc_lm;
            for ( int j=0; j< dv_tf.size();  j++)
            {
                 doc_tf_idf.push_back( ( 1.0+ log(dv_tf[j]))*log( 1.0 + (double)docCountTotal/(double)docCount ));
                  //LM version 
                 double docLen = dv_doc_length[j];
                 double docProb = dv_tf[j]/docLen;
				 double lambda = docLen/double(docLen + iDirPrior);
				 double score = lambda*docProb + (1-lambda)*((double)(termCount)/(double)(totalTermCount));
				 doc_lm.push_back( log(score) );
            }
            
            vect_tf_idf2[i]=Strdev( doc_tf_idf );//vect_tf_idf2.push_back( Strdev( doc_tf_idf ));
            vect_lm2[i]=Strdev( doc_lm );//vect_lm2.push_back( Strdev( doc_lm ));
            i++;
        }
      

       sqc_sum_tf.push_back( Sum( vect_tf_idf1 ));
       sqc_avg_tf.push_back( Average( vect_tf_idf1) ); 
       sqc_max_tf.push_back( *max_element( vect_tf_idf1.begin(), vect_tf_idf1.end() ) );
       sqc_sum_lm.push_back( Sum( vect_lm1 ) );
       sqc_avg_lm.push_back( Average( vect_lm1 ) ); 
       sqc_max_lm.push_back( *max_element( vect_lm1.begin(), vect_lm1.end() ) );
       sum_idf.push_back( Sum( vect_idf ) );
       avg_idf.push_back( Average( vect_idf ) );
       max_idf.push_back( *max_element(  vect_idf.begin(), vect_idf.end()));
 
        var_sum_tf.push_back( Sum( vect_tf_idf2 ));
        var_avg_tf.push_back( Average( vect_tf_idf2 ) );
        var_max_tf.push_back( *max_element( vect_tf_idf2.begin(), vect_tf_idf2.end() ) );
        var_sum_lm.push_back( Sum( vect_lm2 ) );
        var_avg_lm.push_back( Average( vect_lm2 ) );
        var_max_lm.push_back( *max_element( vect_lm2.begin(), vect_lm2.end() ) );    
    }
    r.close();
    
    	WriteQidValueFile("SumSCQTFIDF", qids, sqc_sum_tf );
	WriteQidValueFile("AvgSCQTFIDF", qids,  sqc_avg_tf );
	WriteQidValueFile("MaxSCQTFIDF", qids,   sqc_max_tf );
	WriteQidValueFile("SumSCQLM", qids,  sqc_sum_lm );
	WriteQidValueFile("AvgSCQLM", qids,  sqc_avg_lm );
	WriteQidValueFile("MaxSCQLM", qids,  sqc_max_lm );
	WriteQidValueFile("SumVarTFIDF", qids, var_sum_tf );
	WriteQidValueFile("AvgVarTFIDF", qids, var_avg_tf );
	WriteQidValueFile("MaxVarTFIDF", qids, var_max_tf );
	WriteQidValueFile("SumVarLM", qids, var_sum_lm );
	WriteQidValueFile("AvgVarLM", qids, var_avg_lm );
	WriteQidValueFile("MaxVarLM", qids, var_max_lm );
	WriteQidValueFile("SumIDF",  qids, sum_idf);
	WriteQidValueFile("AvgIDF", qids, avg_idf );
	WriteQidValueFile("MaxIDF", qids, max_idf );

	//load MAP and calculate correlations 
	vector<string> mqids;
	double_vector eval;
	ReadQidValueFile( eval_file, mqids, eval );
	std::set<size_t> ignoreLeft, ignoreRight;
	//mqids can posses less queries then the qids, since the secord is just query file
	MergeQIDs( mqids, qids, ignoreLeft, ignoreRight );
	Filter( eval, ignoreLeft );
	// Filter
	Filter( sqc_sum_tf, ignoreRight );
	Filter( sqc_avg_tf, ignoreRight ); 
	Filter( sqc_max_tf, ignoreRight ); 
	Filter( sqc_sum_lm, ignoreRight ); 
	Filter( sqc_avg_lm, ignoreRight );
	Filter( sqc_max_lm, ignoreRight );
	Filter( var_sum_tf, ignoreRight ); 
	Filter( var_avg_tf, ignoreRight ); 
	Filter( var_max_tf, ignoreRight ); 
	Filter( var_sum_lm, ignoreRight );
	Filter( var_avg_lm, ignoreRight );
	Filter( var_max_lm, ignoreRight ); 
	Filter( sum_idf, ignoreRight ); 
	Filter( avg_idf, ignoreRight ); 
	Filter( max_idf, ignoreRight );
    //output correlations to file
	cout << "SumSCQTFIDF: "<< Correl(  sqc_sum_tf, eval ) << " " << KendallsTau( sqc_sum_tf, eval ) << endl;
	cout << "AvgSCQTFIDF: "<< Correl(  sqc_avg_tf, eval ) << " " << KendallsTau( sqc_avg_tf, eval ) << endl;
	cout << "MaxSCQTFIDF: "<< Correl(  sqc_max_tf, eval ) << " " << KendallsTau( sqc_max_tf, eval ) << endl;
	cout << "SumSCQLM: "<< Correl(  sqc_sum_lm, eval ) << " " << KendallsTau( sqc_sum_lm, eval ) << endl;
	cout << "AvgSCQLM: "<< Correl(  sqc_avg_lm, eval ) << " " << KendallsTau( sqc_avg_lm, eval ) << endl;
	cout << "MaxSCQLM: "<< Correl(  sqc_max_lm, eval ) << " " << KendallsTau( sqc_max_lm, eval ) << endl;
	cout << "SumVarTFIDF: "<< Correl(  var_sum_tf, eval ) << " " << KendallsTau( var_sum_tf, eval ) << endl;
	cout << "AvgVarTFIDF: "<< Correl(  var_avg_tf, eval ) << " " << KendallsTau( var_avg_tf, eval ) << endl;
	cout << "MaxVarTFIDF: "<< Correl(  var_max_tf, eval ) << " " << KendallsTau( var_max_tf, eval ) << endl;
	cout << "SumVarLM: "<< Correl(  var_sum_lm, eval ) << " " << KendallsTau( var_sum_lm, eval ) << endl;
	cout << "AvgVarLM: "<< Correl(  var_avg_lm, eval ) << " " << KendallsTau( var_avg_lm, eval ) << endl;
	cout << "MaxVarLM: "<< Correl(  var_max_lm, eval ) << " " << KendallsTau( var_max_lm, eval ) << endl;
	cout << "SumIDF: "<< Correl( sum_idf , eval ) << " " << KendallsTau( sum_idf, eval ) << endl;
	cout << "AvgIDF: "<< Correl( avg_idf , eval ) << " " << KendallsTau( avg_idf, eval ) << endl;
	cout << "MaxIDF: "<< Correl( max_idf, eval ) << " " << KendallsTau( max_idf, eval ) << endl;
    
	env.close();
    return 0;
  } catch( lemur::api::Exception& e ) {
    LEMUR_ABORT(e);
  }
}


