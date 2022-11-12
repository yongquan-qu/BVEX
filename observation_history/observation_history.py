"""observation_history dataset."""
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import os

# TODO(observation_history): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(observation_history): BibTeX citation
_CITATION = """
"""


class ObservationHistory(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for observation_history dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(highres_forcing): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'states': tfds.features.Tensor(dtype=tf.float32, shape=(None,64,64,2)),
            'time': tfds.features.Tensor(dtype=tf.float32, shape=(None,)),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=None,  # Set to `None` to disable
        homepage='https://github.com/MetLab-HKUST/BVEX',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(highres_forcing): Downloads the data and defines the splits
    path = '/workspace/yquai/BVEX/Data/Train/'

    # TODO(highres_forcing): Returns the Dict[split names, Iterator[Key, Example]]
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                "path": path + "Observation_Train"
            },
        ),
    ]

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(highres_forcing): Yields (key, example) tuples from the dataset
    list_dir = os.listdir(path+'/Physical_States/')
    num_samples = len(list_dir)
    
    
    for i in range(num_samples-1):
        yield i, {
            'states': np.load(f'{path}/Physical_States/{i}.npy').astype('float32'),
            'time': np.load(f'{path}/Time/{i}.npy').astype('float32')
        }