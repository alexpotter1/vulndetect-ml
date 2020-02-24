#!/usr/bin/env python3
'''Adapted from the TensorFlow dataset example for IMDB: https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/imdb.py'''

try:
    import tensorflow_datasets.public_api as tfds
except ModuleNotFoundError:
    print("Tensorflow dataset module not installed. Installing now...")

    from setuptools.command.easy_install import main as install
    install(['tensorflow_datasets'])
    print("Installed!\n")

import os
import re
import util
import parse_training_input

tfds.download.add_checksums_dir(os.getcwd() + os.path.sep + "url_checksums")


class NISTJulietJavaConfig(tfds.core.BuilderConfig):
    @tfds.core.disallow_positional_args
    def __init__(self, text_encoder_config=None, **kwargs):
        super(NISTJulietJavaConfig, self).__init__(
            version=tfds.core.Version('1.0.0'),
            supported_versions=[
                tfds.core.Version(
                    "0.1.0", experiments={tfds.core.Experiment.S3: False}
                )
            ],
            **kwargs)
        
        self.text_encoder_config = (
            text_encoder_config or tfds.features.text.TextEncoderConfig()
        )


ARCHIVE_DOWNLOAD_URL = 'https://www.dropbox.com/s/cu9p16hwoo1yofi/NIST_Juliet_1.3_testcases.zip?dl=1'


class NISTJulietJava(tfds.core.GeneratorBasedBuilder):
    '''TensorFlow dataset for vulnerable Java code in the NIST Juliet core.'''

    VERSION = tfds.core.Version('0.1.0')

    BUILDER_CONFIGS = [
        NISTJulietJavaConfig(
            name="plain_text",
            description="Plain text",
        ),
        NISTJulietJavaConfig(
            name="bytes",
            description=("Uses byte-level text encoding with a ByteTextEncoder"),
            text_encoder_config=tfds.features.text.TextEncoderConfig(encoder=tfds.features.text.ByteTextEncoder()),
        ),
        NISTJulietJavaConfig(
            name="subwords16k",
            description=("Uses a SubwordTextEncoder with 16k vocabulary size"),
            text_encoder_config=tfds.features.text.TextEncoderConfig(
                encoder_cls=tfds.features.text.SubwordTextEncoder,
                vocab_size=2**14
            ),
        )
    ]

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description="This is the dataset for vulnerable Java code defined by NIST. It contains over 100 samples of code vulnerable to a known CWE, and gives examples of vulnerable code and ways to fix.",
            features=tfds.features.FeaturesDict({
                "code": tfds.features.Text(encoder_config=self.builder_config.text_encoder_config),
                "label": tfds.features.ClassLabel(names=util.get_vulnerability_categories())
            }),
            supervised_keys=("code", "label"),
            homepage='https://samate.nist.gov/SRD',
        )

    def _vocab_text_gen(self, archive, directory):
        for _, sample in self._generate_examples(archive, directory):
            yield sample["code"]
    
    def _split_generators(self, dl_manager):
        archive_path = dl_manager.download(ARCHIVE_DOWNLOAD_URL)

        def archive():
            return dl_manager.iter_archive(archive_path)

        self.info.features["code"].maybe_build_from_corpus(self._vocab_text_gen(archive(), 'testcases'))

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                num_shards=10,
                gen_kwargs={
                    "archive": archive(),
                    "directory": 'testcases'
                },
            ),
        ]

    def _generate_examples(self, archive, directory):
        '''Generates code sample text'''

        reg_path = "(?<=-)(.*)(?=.java)"
        
        for path, sample_f in archive:
            res = re.search(reg_path, path)
            if not res:
                continue

            code = parse_training_input.extract_bad_function_from_text(sample_f.read())
            if len(code) == 0:
                # didn't extract anything from 'bad' method, sample is useless?
                continue

            label = res.group(1)

            yield path, {
                "code": code,
                "label": label,
            }
