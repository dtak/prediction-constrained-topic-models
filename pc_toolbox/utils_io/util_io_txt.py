import numpy as np
import codecs

def load_list_of_unicode_from_txt(txt_path):
    ''' Loads list of unicode_strings from plain-txt file

    Returns
    -------
    list_of_u : list of unicode strings
    '''
    possible_fmts = ['utf-8', 'iso-8859-1', 'ascii']
    for fid, fmt in enumerate(possible_fmts):
        try:
            list_of_u = list()
            with codecs.open(txt_path, 'rU', fmt) as f:
                for line in f.readlines():
                    u = line.strip()
                    u = u.replace(u' ', u'_')
                    u = u.replace(u',', u'+')
                    list_of_u.append(u)
            return list_of_u
        except UnicodeDecodeError as e:
            pass
    raise e

def load_list_of_strings_from_txt(txt_path):
    ''' Load list of strings from a plain-txt file

    Assumes each string is on separate line of the file.

    Returns
    -------
    list_of_str : list of strings
        Will have any whitespace replaced by underscore
    '''
    array_of_str = np.loadtxt(txt_path, dtype=str, delimiter='\n')
    if array_of_str.ndim < 1:
        array_of_str = np.expand_dims(array_of_str, axis=0)
    list_of_str = [
        s.replace(' ', '_').replace(',','+')
            for s in array_of_str.tolist()]
    assert isinstance(list_of_str, list)
    return list_of_str
