from functools import reduce

import numpy as np
# from qpp_poc import Index, QueryParser
import warnings
from qpp_poc.qpp_poc import Index, QueryParser, Config


# TODO: OOV terms should be treated differently, check if entire query OOV vs part of it
def generate_matching_terms_dict(candidate_set: dict) -> set:
    # x = reduce(set.intersection, [set(v.keys()) for k, v in candidate_set.items()])
    return reduce(set.union, [set(v.keys()) for k, v in candidate_set.items()])


class LocalManager:
    def __init__(self, index_obj: Index, query_obj: QueryParser, qid):
        self.mu = Config.MU
        self.num_docs = Config.NUM_DOCS
        self.query = query_obj.get_query(qid)
        self.index = index_obj

    def get_matching_postings(self):
        candidate_dict = {}
        for term in self.query:
            term, cf, df, posting_list = self.index.get_posting_list(term)
            candidate_dict[term] = dict(posting_list)
        return candidate_dict

    def score_document(self, doc_id, candidate_dict, doc_len):
        result = 0
        lambda_ = doc_len / (doc_len + self.mu)
        for term, tf_q in self.query.items():
            if self.index.get_term_cf(term) == 0:  # Workaround for OOV query terms
                print(f'{term} is OOV !!')
                continue
            result += np.log(
                lambda_ * candidate_dict[term].get(doc_id, 0) / doc_len + (1 - lambda_) * self.index.get_term_cf(
                    term) / self.index.total_terms) * tf_q
            # _temp = np.log(
            #     lambda_ * candidate_dict[term].get(doc_id, 0) / doc_len + (1 - lambda_) * self.index.get_term_cf(
            #         term) / self.index.total_terms) * tf_q
            # if _temp == -np.inf:
            #     print(term)
            # result += _temp
        return result

    def score_candidates(self, candidate_set, candidate_dict):
        result_set = {}
        for doc_id in candidate_set:
            result_set[doc_id] = self.score_document(doc_id, candidate_dict, self.index.get_doc_len(doc_id))
        return result_set

    def translate_doc_id_to_doc_no(self, result_tuple):
        return tuple(map(lambda x: (self.index.get_doc_name(x[0]), x[1]), result_tuple))

    def run_retrieval(self):
        candidate_dict = self.get_matching_postings()
        candidate_set = generate_matching_terms_dict(candidate_dict)
        scored_dict = self.score_candidates(candidate_set, candidate_dict)
        return self.translate_doc_id_to_doc_no(
            sorted(scored_dict.items(), key=lambda kv: kv[1], reverse=True)[:self.num_docs])
