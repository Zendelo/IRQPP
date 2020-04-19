import math
import numpy as np

from qpp_poc.qpp_poc import Index, QueryParser


class LocalManager:
    def __init__(self, index_obj: Index, query_obj: QueryParser, qid):
        self.mu = 2500
        self.query = query_obj.get_query(qid)
        self.index = index_obj

    def get_matching_postings(self):
        candidate_set = []
        for term in self.query:
            candidate_set.append(self.index.get_posting_list(term))
        return candidate_set

    def score_document(self, query, doc_len, terms_tf):
        result = -np.inf
        lambda_ = doc_len / (doc_len + self.mu)
        for term, tf_q in query.items():
            result += np.log(lambda_ * terms_tf.get(term, 0) + (1 - lambda_) * self.index.get_term_cf(
                term) / self.index.total_terms) * tf_q
        return result

    def score_candidates(self, qry_dict, candidate_set):
        scored_set = candidate_set * query_dict
        return candidate_set

    def local_manager(self, qry_dict):
        candidate_set = self.run_matching()
        scored_set = self.score_candidates(qry_dict, candidate_set)
        return scored_set


if __name__ == '__main__':
    query_dict = {}
    LocalManager(query_dict)
