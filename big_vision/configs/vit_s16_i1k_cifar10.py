# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=line-too-long
r"""Pre-training ViT-S/16 on ILSVRC-2012 following https://arxiv.org/abs/2205.01580.

This should take 6-7h to finish 90ep on a TPU-v3-8 and reach 76.5%,
see the tech report for more details.

Command to run:

big_vision.train \
    --config big_vision/configs/vit_s16_i1k.py \
    --workdir gs://[your_bucket]/big_vision/`date '+%m-%d_%H%M'`

To run for 300ep, add `--config.total_epochs 300` to the command.
"""

import ml_collections as mlc


def get_config():
  """Config for training."""
  config = mlc.ConfigDict()

  config.seed = 0
  config.total_epochs = 200
  config.num_classes = 10
  config.loss = 'softmax_xent'

  config.input = {}
  config.input.data = dict(
      name='cifar10',
      split='train[:90%]',
  )
  config.input.batch_size = 256
  config.input.cache_raw = True  # Needs up to 120GB of RAM!
  config.input.shuffle_buffer_size = 50_000

  config.input.pp = (
      'decode|resize(32)|flip_lr|value_range(-1, 1)|onehot(10, key="label", key_result="labels")|keep("image", "labels")'
  )
  pp_eval = 'decode|resize(32)|value_range(-1, 1)|onehot(10, key="label", key_result="labels")|keep("image", "labels")'

  # To continue using the near-defunct randaug op.
  config.pp_modules = ['ops_general', 'ops_image', 'ops_text', 'archive.randaug']

  config.log_training_steps = 50
  config.ckpt_steps = 1000

  # Model section
  config.model_name = 'vit'
  config.model = dict(
      variant='S/16',
      rep_size=True,
      pool_type='gap',
      posemb='sincos2d',
  )

  # Optimizer section
  config.grad_clip_norm = 1.0
  config.optax_name = 'scale_by_adam'
  config.optax = dict(mu_dtype='bfloat16')

  config.lr = 0.01
  config.wd = 0.0001
  config.schedule = dict(warmup_steps=500, decay_type='cosine')

  config.mixup = dict(p=0.2, fold_in=None)

  # Eval section
  def get_eval(split, dataset='cifar10'):
    return dict(
        type='classification',
        data=dict(name=dataset, split=split),
        pp_fn=pp_eval.format(lbl='label'),
        loss_name=config.loss,
        log_steps=2500,  # Very fast O(seconds) so it's fine to run it often.
    )
  config.evals = {}
  config.evals.train = get_eval('train[:90%]')
  config.evals.val = get_eval('train[90%:]')
  config.evals.test = get_eval('test')

  return config
