"""
Write all lower-case environment variables to stdout

Examples
--------
>>> os.environ['abc'] = '123'
>>> os.environ['DO_NOT_PRINT'] = '456'
>>> print_lowercase_env_vars_as_keyword_args()
--abc 123

"""

from __future__ import print_function

import os

def print_lowercase_env_vars_as_keyword_args():
    try:
        XHOST_PREFIXES_TO_SKIP = os.environ['XHOST_PREFIXES_TO_SKIP'].split(',')
    except KeyError:
        XHOST_PREFIXES_TO_SKIP = list()
    XHOST_PREFIXES_TO_SKIP = set(XHOST_PREFIXES_TO_SKIP)

    for key in sorted(os.environ.keys()):
        if key[0].islower():
            val = os.environ[key]
            # Manually remove some unnecessary env vars
            if key in XHOST_PREFIXES_TO_SKIP:
                continue

            if key == 'extra_kwargs_str':
                # Handle args like "name1,val1,name2,val2"
                kkeys = val.split(",")[::2]
                kvals = val.split(",")[1::2]
                for k, v in zip(kkeys,kvals):
                    if not k.startswith("--"):
                        k = "--" + k
                    print("%s %s" % (k, v))
                continue

            # handle negative nums as values
            if val.startswith("-") and val[1].isdigit():
                val = '" %s"' % (val)
            # handle paths with ( or ) in them
            if val.count("("):
                # wrap parentheses with double quotes
                val = '"%s"' % (val)
            if val.count(" ") and not (val.startswith('"') and val.endswith('"')):
                val = '"%s"' % (val)

            #else:
            #    assert val.count(" ") == 0

            assert key.count(" ") == 0
            print("--%s %s" % (key, val))

if __name__ == '__main__':
    print_lowercase_env_vars_as_keyword_args()
