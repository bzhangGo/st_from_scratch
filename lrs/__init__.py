# coding: utf-8

from lrs import noamlr, epochlr, cosinelr


def get_lr(params):

    strategy = params.lrate_strategy.lower()

    if strategy == "noam":
        return noamlr.NoamDecayLr(
            params.lrate,
            params.min_lrate,
            params.max_lrate,
            params.warmup_steps,
            params.hidden_size
        )
    elif strategy == "epoch":
        return epochlr.EpochDecayLr(
            params.lrate,
            params.min_lrate,
            params.max_lrate,
            params.lrate_decay,
        )
    elif strategy == "cosine":
        return cosinelr.CosineDecayLr(
            params.lrate,
            params.min_lrate,
            params.max_lrate,
            params.warmup_steps,
            params.lrate_decay,
            t_mult=params.cosine_factor,
            update_period=params.cosine_period
        )
    else:
        raise NotImplementedError(
            "{} is not supported".format(strategy))
