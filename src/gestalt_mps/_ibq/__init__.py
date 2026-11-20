"""Small inference-only port of the IBQ tokenizer (SEED-Voken, Apache License 2.0).

Only the encoder, decoder, ``quant_conv``, ``post_quant_conv`` and the codebook are
reproduced. State-dict names match the original ``IBQ`` Lightning module so that the
official checkpoint maps one-to-one. See ``THIRD_PARTY.md`` for provenance.
"""
