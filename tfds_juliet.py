#!/usr/bin/env python3

try:
    import tensorflow_datasets.public_api as tfds
except ModuleNotFoundError:
    print("Tensorflow dataset module not installed. Installing now...")

    from setuptools.command.easy_install import main as install
    install(['tensorflow_datasets'])
    print("Installed!\n")


class NISTJulietJavaDataset(tfds.core.GeneratorBasedBuilder):
    '''TensorFlow dataset for vulnerable Java code in the NIST Juliet core.'''

    VERSION = tfds.core.Version('0.1.0')

    def _info(self):
        pass
    
    def _split_generators(self, dl_manager):
        pass

    def _generate_examples(self):
        pass
