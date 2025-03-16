import ml_collections as mlc


def get_config():
    """Config for training."""
    config = mlc.ConfigDict()

    config.seed = 0
    config.total_epochs = 200
    config.num_classes = 100
    config.loss = 'softmax_xent'

    config.input = {}
    config.input.data = dict(
        name='cifar100',
        split='train[:90%]',
    )
    config.input.batch_size = 256
    config.input.cache_raw = True
    config.input.shuffle_buffer_size = 50_000

    config.input.pp = (
        'decode|resize(32)|flip_lr|value_range(-1, 1)|onehot(100, key="label", key_result="labels")|keep("image", "labels")'
    )
    pp_eval = 'decode|resize(32)|value_range(-1, 1)|onehot(100, key="label", key_result="labels")|keep("image", "labels")'

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
    def get_eval(split, dataset='cifar100'):
        return dict(
            type='classification',
            data=dict(name=dataset, split=split),
            pp_fn=pp_eval.format(lbl='label'),
            loss_name=config.loss,
            log_steps=2500,
        )

    config.evals = {}
    config.evals.train = get_eval('train[:90%]')
    config.evals.val = get_eval('train[90%:]')
    config.evals.test = get_eval('test')

    return config
