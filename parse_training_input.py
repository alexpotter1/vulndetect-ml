#!/usr/bin/env python3

import javalang
import numpy as np
import re
import os
import util
import ray
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def isPrimitive(obj):
    return not hasattr(obj, '__dict__')

def extract_bad_function(file_path):
    return extract_function(file_path, criterion='bad')

def extract_function(file_path, criterion):
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
            except:
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

    with open(file_path, 'r') as file:
        src = file.read()
        src_split = src.split('\n')

        try:
            tree = javalang.parse.parse(src)
            for _, node in tree.filter(javalang.tree.MethodDeclaration):
                if node.name.lower() == str(criterion).lower():
                    # tokenise, find the start/end of method, and extract from the file
                    # needed since javalang can't convert back to java src
                    start_pos = node.position.line
                    end_pos = None
                    if (len(node.body) > 0):
                        end_pos = recursive_find_deepest_child_position(node.body[-1])

                    if end_pos is None:
                        end_pos = start_pos

                    function_text = ""
                    for line in range(start_pos, end_pos+1):
                        function_text += src_split[line-1]
                    
                    return function_text
                
            return None
        except (javalang.parser.JavaSyntaxError, javalang.parser.JavaParserError) as e:
            print("ERROR OCCURRED DURING PARSING")
            print(e)

def chunkstring(string, length):
    return (string[0+i:length+i] for i in range(0, len(string), length))

def vectorise(body, vecsize=util.VEC_SIZE, norm_wspace=True):
    if body is None or len(body) == 0:
        return None

    if norm_wspace:
        body = body.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        body = re.sub(r'\s+', ' ', body)
    
    # trim to maximum vector size if body is super duper large
    body = body[0:vecsize]

    # split body into chunks, generate one-hot encoding
    chunks = list(chunkstring(body, util.ONEHOT_CHUNK_SIZE))

    onehot_encoded = util.one_hot_string(chunks, mapping_tuple=util.CHAR_MAPPING, string_chunk_length=util.ONEHOT_CHUNK_SIZE)
      
    '''if len(onehot_encoded) < vecsize:
        for _ in range(0, vecsize - len(onehot_encoded)):
            f_vec.append(util.pad_vector)'''

    return onehot_encoded

@ray.remote
def get_vulnerable_code_samples_for_path(base_path):
    i = 0
    X = []
    Y = []
    for path, _, files in util.walk_level(base_path, level=0):
        if len(files) > 0:
            label = util.trim_root_path(util.BASE_PATH, base_path)
            for f in files[:util.MAX_FILE_PARSE]:
                if (f.lower().endswith('.java')):
                    print("Extracting from %s" % f)
                    # extract 'bad' function, vectorise and add to list
                    result = vectorise(extract_bad_function(path + os.path.sep + f))
                    if result is not None:
                        X.append(result)

                        one_hot = util.one_hot_string(list(label), mapping_tuple=util.LABEL_CHAR_MAPPING, string_chunk_length=1)
                        Y.append(one_hot)
        
            i += 1

    return np.asarray(X), np.asarray(Y)

def save_vulnerable_code_samples(base_path):
    print("Parallelizing...")
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
            print("Saved vector: %s for category %s" % (save_name, categories[processed]))
            processed += 1
            

    ray.shutdown()
    


