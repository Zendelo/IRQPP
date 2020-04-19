import linecache


def get_file_len(file_path):
    """Function that opens a file and counts the number of lines in it"""
    return sum(1 for _ in open(file_path))


def read_line(file_path, n):
    """Function used to return a specific line n from a file, if the line doesn't exist, returns an empty string"""
    return linecache.getline(file_path, n)
