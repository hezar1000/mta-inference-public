import bz2
import pickle

def compress_data(data):
    """
    Helper function: compress object into a BLOB
    """
    return bz2.compress(pickle.dumps(data))

def decompress_data(data_compressed):
    """
    Helper function: decompress BLOB into object
    """
    return pickle.loads(bz2.decompress(data_compressed))

def dump_data(data):
    """
    Helper function: dump data without compressing
    """
    return pickle.dumps(data)

def undump_data(data_dumped):
    """
    Helper function: load dumped data
    """
    return pickle.loads(data_dumped)
