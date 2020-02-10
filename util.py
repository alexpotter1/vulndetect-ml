import numpy as np
from tensorflow.keras.preprocessing.text import one_hot
import os
import glob
import json
import itertools

BASE_PATH = os.getcwd() + "/testcases/"
SAVE_PATH = BASE_PATH + "vectorised/"

# stop memory usage getting hugh mungus
MAX_FILE_PARSE = 75

VEC_SIZE = 4096

supported_char_list = r"abcdefghijklmnopqrstuvwxyz" \
                      r"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" \
                      r"-,;.!?:'/\|_@#$%^&*~`+-=<>()[]{}\" "

supported_char_map = {}
pad_vector = [0 for c in supported_char_list]


def memoize(func):
    from functools import wraps

    memo = {}
    @wraps(func)
    def wrapper(*args):
        if args in memo:
            return memo[args]
        else:
            ret = func(*args)
            memo[args] = ret
            return ret

    return wrapper

# naive, all combinations
@memoize
def generate_char_chunk_mapping(chunk_length=3):
    char_list = list(supported_char_list)
    mapping = {}
    i = 0

    # 1-length mapping
    for char in char_list:
        mapping[char] = i
        i += 1

    if chunk_length > 1:
        for repeat in range(2, chunk_length + 1):
            combinations = [''.join(i) for i in itertools.product(char_list, repeat=repeat)]
            for char in combinations:
                mapping[char] = i
                i += 1

    # return mapped dict and max idx value
    print("Generated chunk mapping with max index: " + str(i))
    return (mapping, i)


def one_hot_string(string_list, mapping_tuple, string_chunk_length=3):
    if string_chunk_length != len(string_list[0]):
        string_chunk_length = len(string_list[0])

    _, max_idx = mapping_tuple

    # generate an one-hot-encoded numpy array
    # from the tensor of the chunked string
    encoded = [one_hot(chunk, max_idx) for chunk in string_list]
    return encoded


ONEHOT_CHUNK_SIZE = 1
CHAR_MAPPING = generate_char_chunk_mapping(ONEHOT_CHUNK_SIZE)


@memoize
def get_vulnerability_categories(base_path=BASE_PATH):
    # populate labels
    vulnerability_categories = []
    for path, _, files in walk_level(base_path, level=1):
        if len(files) > 0:
            label = trim_root_path(base_path, path)
            print("Discovered input directory: %s" % label)

            vulnerability_categories.append(label)

    return set(vulnerability_categories)


def serialise_to_json(object, path):
    with open(path, 'w') as f:
        print("Saving object to ", path)
        json.dump(object, f)


def get_label_category_from_int(number):
    lmap = None
    if not os.path.exists(os.path.join(SAVE_PATH, 'label_map.json')):
        if label_map is not None:
            lmap = label_map
        else:
            raise RuntimeError("Could not find an appropriate label map! No associated JSON file/not in memory")
    else:
        with open(os.path.join(SAVE_PATH, 'label_map.json'), 'r') as f:
            lmap = json.load(f)
    
    matches = [k for k, v in lmap.items() if v == number]
    if len(matches) == 1:
        return matches[0]
    else:
        print("get_label_category_from_int(): Hmm, got multiple matches for int ", number)
        return matches


def walk_level(directory, level=1):
    directory = directory.rstrip(os.path.sep)
    # assert os.path.isdir(directory)
    sep_count = directory.count(os.path.sep)
    for root, dirs, files in os.walk(directory):
        yield root, dirs, files
        sep_count_current = root.count(os.path.sep)
        if sep_count + level <= sep_count_current:
            del dirs[:]


def get_saved_vector_list(read_max='all'):
    vectors = glob.glob(SAVE_PATH + "*.npz")
    if read_max != 'all' and isinstance(int, read_max):
        vectors = vectors[:read_max]

    return vectors


def do_saved_vectors_exist():
    return len(get_saved_vector_list()) > 0


def read_saved_vectors(read_max='all'):
    vectors = get_saved_vector_list(read_max)

    for f_path in vectors:
        with np.load(f_path) as data:
            yield data['X'], data['Y']


def trim_root_path(base_path, path):
    return path.replace(base_path, '')


def _init():
    i = 0
    for c in supported_char_list:
        vec = [0 for ch in supported_char_list]
        vec[i] = 1
        supported_char_map[c] = vec
        i += 1


_init()
labels = get_vulnerability_categories()
label_values = list(range(len(labels)))
label_map = {key: value for key, value in zip(labels, label_values)}
serialise_to_json(label_map, os.path.join(SAVE_PATH, 'label_map.json'))
