import numpy as np
from torch.optim.lr_scheduler import LambdaLR


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
