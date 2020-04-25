import qpp_poc.qpp_poc.CommonIndexFileFormat_pb2 as ciff
from google.protobuf.internal.decoder import _DecodeVarint32
import sys
from Timer import timer
from qpp_poc.qpp_poc import Config, read_line, get_file_len, TermPosting, Posting, TermRecord, DocRecord

"""
An index stored in CIFF is a single file comprised of exactly the following:
    - A Header protobuf message,
    - Exactly the number of PostingsList messages specified in the num_postings_lists field of the Header
    - Exactly the number of DocRecord messages specified in the num_doc record field of the Header
 """

INDEX_CIFF_FILE = "/research/local/olz/ciff_indexes/robust04-Lucene-indri-krovetz.ciff"


def parse_posting_list(posting_list: ciff.PostingsList) -> TermPosting:
    if posting_list:
        if not isinstance(posting_list, ciff.PostingsList):
            raise TypeError(f'parse_posting_lists is expecting a ciff.PostingsList {type(posting_list)} was passed')
    term, cf_t, df_t, posting_list = posting_list.term, posting_list.cf, posting_list.df, posting_list.postings
    return TermPosting(term, cf_t, df_t, tuple(Posting(p.docid, p.tf) for p in posting_list))


def read_message(buffer, n, message_type):
    message = message_type()
    msg_len, new_pos = _DecodeVarint32(buffer, n)
    n = new_pos
    msg_buf = buffer[n:n + msg_len]
    n += msg_len
    message.ParseFromString(msg_buf)
    return n, message


def record_positions(buffer, beginning):
    msg_len, new_pos = _DecodeVarint32(buffer, beginning)
    return new_pos, new_pos + msg_len


@timer
def parse_index_file(index_file):
    terms_dict = {}
    doc_records = {}
    with open(index_file, 'rb') as fp:
        buf = fp.read()
        n = 0
        cur_n, header = read_message(buf, n, ciff.Header)
        num_postings_lists = header.num_postings_lists
        for _ in range(num_postings_lists):
            n, _posting_list = read_message(buf, cur_n, ciff.PostingsList)
            terms_dict[_posting_list.term] = TermRecord(_posting_list.term, cur_n, _posting_list.cf, _posting_list.df)
            cur_n = n
        num_doc_records = header.num_docs
        for _ in range(num_doc_records):
            n, _doc_record = read_message(buf, cur_n, ciff.DocRecord)
            doc_records[_doc_record.docid] = DocRecord(cur_n, _doc_record.collection_docid, _doc_record.doclength)
            cur_n = n
    return header, terms_dict, doc_records


class Index:
    @classmethod
    def oov(cls, term):
        return f"{term} 0 0"  # Out of vocabulary terms

    def __init__(self, index_file, header, terms_dict, doc_records):
        with open(index_file, 'rb') as fp:
            self.file_buf = fp.read()
        self.header = header
        self.terms_dict = terms_dict
        self.doc_records = doc_records

    def _get_raw_posting_list(self, term):
        term_id = self.terms_dict.get(term)
        if term_id:
            return self._read_index_line(term_id.id)
        else:
            return Index.oov(term)

    def _read_index_line(self, n):
        _, posting_list = read_message(self.file_buf, n, ciff.PostingsList)
        return posting_list

    def get_posting_list(self, term: str) -> TermPosting:
        posting_lists = self._get_raw_posting_list(term)
        return parse_posting_list(posting_lists)

    def get_doc_len(self, doc_id):
        return self.doc_records.get(doc_id).doc_len

    def get_doc_name(self, doc_id):
        return self.doc_records.get(doc_id).collection_doc_id


@timer
def main():
    index = Index(INDEX_CIFF_FILE, header, terms_dict, doc_records)
    x = index.get_posting_list('ingathering')
    print(len(x.posting_list))


if __name__ == '__main__':
    header, terms_dict, doc_records = parse_index_file(INDEX_CIFF_FILE)
    main()
