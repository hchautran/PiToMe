import torch
import ml_collections


import numpy as np
from torch.optim.lr_scheduler import LambdaLR
import os


DATA_PATH = f'{os.getcwd()}/data/tc' #you can change this to your desired path


# kindly adapted from google-research/long-range-arena code
def create_learning_rate_scheduler(factors, config):
    """
      Creates learning rate schedule.
      Interprets factors in the factors string which can consist of:
      * constant: interpreted as the constant value,
      * linear_warmup: interpreted as linear warmup until warmup_steps,
      * rsqrt_decay: divide by square root of max(step, warmup_steps)
      * rsqrt_normalized_decay: divide by square root of max(step/warmup_steps, 1)
      * decay_every: Every k steps decay the learning rate by decay_factor.
      * cosine_decay: Cyclic cosine decay, uses steps_per_cycle parameter.
      Args:
        factors: string, factors separated by '*' that defines the schedule.
        config:
            config.learning_rate: float, the starting constant for the lr schedule.
            config.warmup_steps: int, how many steps to warm up for in the warmup schedule.
            config.decay_factor: float, the amount to decay the learning rate by.
            config.steps_per_decay: int, how often to decay the learning rate.
            config.steps_per_cycle: int, steps per cycle when using cosine decay.
      Returns:
        a function of signature optimizer->(step->lr).
  """
    factors = [n.strip() for n in factors.split('*')]
    base_learning_rate: float = config.learning_rate
    warmup_steps: int = config.get('warmup_steps', 1000)
    decay_factor: float = config.get('decay_factor', 0.5)
    steps_per_decay: int = config.get('steps_per_decay', 20000)
    steps_per_cycle: int = config.get('steps_per_cycle', 100000)

    def step_fn(step):
        """ Step to learning rate function """
        ret = 1.0
        for name in factors:
            if name == 'constant':
                ret *= base_learning_rate
            elif name == 'linear_warmup':
                ret *= np.minimum(1.0, step / warmup_steps)
            elif name == 'rsqrt_decay':
                ret /= np.sqrt(np.maximum(step, warmup_steps))
            elif name == 'rsqrt_normalized_decay':
                ret *= np.sqrt(warmup_steps)
                ret /= np.sqrt(np.maximum(step, warmup_steps))
            elif name == 'decay_every':
                ret *= (decay_factor ** (step // steps_per_decay))
            elif name == 'cosine_decay':
                progress = np.maximum(0.0, (step - warmup_steps) / float(steps_per_cycle))
                ret *= np.maximum(0.0, 0.5 * (1.0 + np.cos(np.pi * (progress % 1.0))))
            else:
                raise ValueError('Unknown factor %s.' % name)
        return ret

    return lambda optimizer: LambdaLR(optimizer, step_fn)

# helper fns
def make_char_tokenizer(allowed_chars, lowercase_input=False):
    # make distinct
    allowed_chars = list(set(allowed_chars))

    def _tokenizer(x, max_length):
        # note: x is not batched
        x = x[:max_length]
        if lowercase_input:
            x = x.lower()
        n = len(x)
        mask = ([1] * n) + ([0] * (max_length - n))
        ids = list(map(lambda c: allowed_chars.index(c) + 1, x)) + ([0] * (max_length - n))
        return {'input_ids': torch.LongTensor([ids]), 'attention_mask': torch.LongTensor([mask])}

    _tokenizer.vocab_size = len(allowed_chars) + 1
    return _tokenizer


def make_word_tokenizer(allowed_words, lowercase_input=False, allow_unk=True):
    # make distinct
    allowed_words = list(set(allowed_words))
    PAD_TOKEN = 0
    UNK_TOKEN = 1

    def _tokenizer(x_str, max_length):
        # note: x_str is not batched
        if lowercase_input:
            x_str = x_str.lower()

        x = x_str.split()
        x = x[:max_length]
        n = len(x)
        mask = ([1] * n) + ([0] * (max_length - n))
        ids = list(map(lambda c: allowed_words.index(c) + 2 if c in allowed_words else UNK_TOKEN, x)) + \
                  ([PAD_TOKEN] * (max_length - n))
        if not allow_unk:
            assert UNK_TOKEN not in ids, "unknown words are not allowed by this tokenizer"
        return {'input_ids': torch.LongTensor([ids]), 'attention_mask': torch.LongTensor([mask])}

    _tokenizer.vocab_size = len(allowed_words) + 2
    return _tokenizer



ascii_tokenizer = make_char_tokenizer(''.join(chr(i) for i in range(256)))




def get_text_classification_config(num_labels=2):
    config = ml_collections.ConfigDict()
    config.batch_size = 4
    config.eval_frequency = 100
    config.total_train_samples = 640000
    config.total_eval_samples = -1
    config.learning_rate = 0.0001
    config.weight_decay = 1e-1
    config.warmup_steps = 8000
    config.lr_scheduler = create_learning_rate_scheduler("constant * linear_warmup * cosine_decay", config)
    config.tokenizer = ascii_tokenizer
    config.tied_weights = False
    config.max_length = 1000

    model_config = ml_collections.ConfigDict()
    model_config.max_position_embeddings = config.max_length
    model_config.num_attention_heads = 4
    model_config.num_hidden_layers = 4
    model_config.hidden_size = 256
    model_config.intermediate_size = 1024
    model_config.num_labels = num_labels
    model_config.vocab_size = config.tokenizer.vocab_size

    return config, model_config


