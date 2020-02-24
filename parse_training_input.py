#!/usr/bin/env python3

import javalang
import numpy as np
from scipy import sparse
import os
import util
import ray

try:
    import tensorflow_datasets.public_api as tfds
except ModuleNotFoundError:
    print("Tensorflow dataset module not installed. Installing now...")

    from setuptools.command.easy_install import main as install
    install(['tensorflow_datasets'])
    print("Installed!\n")


def isPrimitive(obj):
    return not hasattr(obj, '__dict__')


def extract_bad_function_from_text(src):
    return extract_function_from_text(src, criterion='bad')


def extract_function_from_text(src, criterion='bad'):
    def recursive_find_deepest_child_position(node_body, prev_deepest=0):
        child_direct_child_set = None

        # line number, don't currently care about column too much
        if isinstance(node_body, list):
            deepest_position = prev_deepest
            node_children = [c for c in node_body if c is not isPrimitive(c) and c is not None]

            if len(node_children) == 0:
                return deepest_position
        else:
            if node_body.position is not None:
                deepest_position = node_body.position.line
            else:
                deepest_position = prev_deepest
            node_children = [c for c in node_body.children if c is not isPrimitive(c) and c is not None]

            if len(node_children) == 0:
                return deepest_position

        for child in node_children:
            try:
                if child.position is not None:
                    child_sub_pos = child.position.line
                else:
                    child_sub_pos = deepest_position

                child_direct_child_set = child.children
            except AttributeError:
                # most likely is not an object
                child_sub_pos = deepest_position
                if isinstance(child, list):
                    child_direct_child_set = child
                else:
                    child_direct_child_set = []

            if len(child_direct_child_set) > 0:
                child_sub_pos = recursive_find_deepest_child_position(child_direct_child_set, prev_deepest=child_sub_pos)

            if child_sub_pos > deepest_position:
                deepest_position = child_sub_pos

        return deepest_position

    src_split = src.decode('utf-8')
    try:
        tree = javalang.parse.parse(src)
        for _, node in tree.filter(javalang.tree.MethodDeclaration):
            if node.name.lower() == str(criterion).lower():
                # tokenise, find the start/end of method,
                # and extract from the file
                # needed since javalang can't convert back to java src
                start_pos = node.position.line
                end_pos = None
                if (len(node.body) > 0):
                    end_pos = recursive_find_deepest_child_position(node.body[-1])

                if end_pos is None:
                    end_pos = start_pos

                function_text = ""
                for line in range(start_pos, end_pos + 1):
                    function_text += src_split[line - 1]

                return function_text

        return ""
    except (javalang.parser.JavaSyntaxError,
            javalang.parser.JavaParserError) as e:
        print("ERROR OCCURRED DURING PARSING")
        print(e)


def extract_bad_function(file_path):
    return extract_function(file_path, criterion='bad')


def extract_function(file_path, criterion):
    with open(file_path, 'r') as f:
        return extract_function_from_text(f.read(), criterion)


def chunkstring(string, length):
    return (string[0 + i:length + i] for i in range(0, len(string), length))


def vectorise_texts(texts, vecsize=util.VEC_SIZE):
    print("TEXTS LENGTH: " + str(len(texts)))
    if texts is None or len(texts) == 0:
        return None

    def flatten(lst):
        return [item for sublist in lst for item in sublist]

    def right_zeropad(lst):
        lst += [0] * (vecsize - len(lst))
        return lst

    text_encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(texts, len(flatten(texts)))
    print("Vocabulary size: %s" % text_encoder.vocab_size)

    encoded = np.zeros((len(texts), vecsize), dtype=np.int32)
    i = 0
    for text in texts:
        encoded[i] = right_zeropad(text_encoder.encode(text))
        i += 1
    
    return sparse.csr_matrix(encoded)


@ray.remote
def get_vulnerable_code_samples_for_path(base_path):
    i = 0
    texts = []
    labels = []
    label_map = util.label_map
    for path, _, files in util.walk_level(base_path, level=0):
        if len(files) > 0:
            label = util.trim_root_path(util.BASE_PATH, base_path)
            for f in files[:util.MAX_FILE_PARSE]:
                if (f.lower().endswith('.java')):
                    print("Extracting from %s" % f)
                    # collate extracted texts then tokenise/one-hot
                    result = extract_bad_function(path + os.path.sep + f)
                    if result is not None:
                        texts.append(result)
                        # integer encode this so it's reversible
                        label_idx = label_map[label]
                        one_hot_label_vector = [0] * int(util.VEC_SIZE / util.VEC_MULTIPLIER)
                        one_hot_label_vector[label_idx] = 1
                        labels.append(one_hot_label_vector)
            i += 1

    vector_texts = vectorise_texts(texts)
    return vector_texts, np.asarray(labels)


def save_vulnerable_code_samples(base_path):
    print("Saving vectorised form of code samples...")
    categories = list(util.get_vulnerability_categories())

    processed = 0
    print("Class count: " + str(len(categories)))
    print("Using npz data format")
    while (processed < len(categories)):
        task_ids = [get_vulnerable_code_samples_for_path.remote(base_path + categories[i]) for i in range(len(categories))]

        for i in range(len(task_ids)):
            ret_X, ret_Y = ray.get(task_ids[i])
            save_name = "%svector-%s" % (util.SAVE_PATH, processed)
            np.savez(save_name, X=ret_X, Y=ret_Y)
            print("Saved vector: %s for category %s" %
                  (save_name, categories[processed]))

            processed += 1

    ray.shutdown()
