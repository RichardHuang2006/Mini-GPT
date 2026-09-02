"""Mini-GPT — a from-scratch GPT pipeline, one module per concept.

Read the modules in the README's order: config, tokenizer, data, model, train,
generate, kernels, posttrain, evaluate. Each is also a CLI entry point where it
makes sense, e.g. ``python -m mini_gpt.train --tier nano ...``.

Nothing is re-exported here on purpose: the package is meant to be read module
by module, and every example imports from the module that teaches the concept.
"""
