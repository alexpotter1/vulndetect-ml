#!/usr/bin/env python3

import javalang


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

    if not isinstance(src, str):
        src = src.decode('utf-8')

    src_split = src.split('\n')
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
