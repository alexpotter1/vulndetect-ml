import os


def get_vulnerability_categories():
    with open(os.path.join(os.getcwd(), 'labels.txt'), 'r') as f:
        vulnerability_categories = [line.rstrip('\n') for line in f]
    
    return vulnerability_categories


def get_label_category_from_int(number):
    labels = get_vulnerability_categories()
    for i, label in enumerate(labels):
        if number == i:
            return label
    
    # return None if no match
    return None
