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


def register_custom_dataset_with_tfds():
    import tensorflow_datasets as tfds
    import shutil

    tfds_path = os.path.join(tfds.__path__[-1], "text")
    import_str = "from tensorflow_datasets.text.tfds_juliet import NISTJulietJava\n"

    print("Registering NIST Juliet dataset with TFDS...")
    shutil.copy2(os.path.join(os.getcwd(), "tfds_juliet", "tfds_juliet.py"), tfds_path)
    with open(os.path.join(tfds_path, '__init__.py'), 'a') as f:
        f.write(import_str)
    
    print("Done! Ensure TFDS is re-imported")
