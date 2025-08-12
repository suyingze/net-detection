import re
import string
from nostril import nonsense


def sanitize_string(s):
    """preprocess string for FastText model training

    Args:
        s (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Translate non-ASCII character codes.
    s = s.strip().encode('ascii', errors='ignore').decode()
    if re.search(r'([0-9\.]*):([0-9]*)->([0-9\.]*):([0-9]*)',s):
        # s = s.replace('/32','')
        split_path = re.split(r'/|\.|,|:|-|>',s)
        split_path = [item for item in filter(lambda x:x != '',split_path)]
        split_path.pop(4)
        split_path.pop(8)
        return split_path
    # Lower-case the string & strip non-alpha.
    for i in s:
        if i in string.punctuation:
            s = s.replace(i," ")

    split_path = s.lower().split()
    # split_path = [item for item in filter(lambda x:x != '',split_path)]
    newline = []
    for item in split_path:
        if len(item) < 2 or item.isdigit():
            continue
        if len(item) <= 5 and len(item) >= 2:
            newline.append(item)
        else:
            try:
                if not nonsense(item):
                    newline.append(item)
                else:
                    newline.append('hash')
            except Exception as e: # error: Text is too short to test
                pass
    split_path = [item for item in filter(lambda x:x != '',newline)]
    return split_path