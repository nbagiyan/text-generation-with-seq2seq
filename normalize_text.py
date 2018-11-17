import unicodedata
import re
import string

def normalize_string(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = ''.join([x for x in s if x not in string.punctuation])
    return s
