import linecache
from bisect import bisect_left


def get_file_len(file_path):
    """Opens a file and counts the number of lines in it"""
    return sum(1 for _ in open(file_path))


def read_line(file_path, n):
    """Return a specific line n from a file, if the line doesn't exist, returns an empty string"""
    return linecache.getline(file_path, n)


def binary_search(list_, target):
    """Return the index of first value equal to target, if non found will raise a ValueError"""
    i = bisect_left(list_, target)
    if i != len(list_) and list_[i] == target:
        return i
    raise ValueError
