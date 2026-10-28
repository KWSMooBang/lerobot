# lerobot_policy_sc_template

A small LIBERO policy to copy when trying an idea (`scripts/shortcut/train/new_policy.sh <name>`).
lerobot out-of-tree plugin (`docs/source/bring_your_own_policies.mdx`): policy type `sc_template`.

```
images (agentview, wrist; resized to 128 px) ──► ConvEncoder per camera (4 stride-2 blocks, FiLM from the instruction) ─┐
state (8) ──► linear ──────────────────────────────────────────────────────────────────────────────────────────────────┼─► MLP ─► action chunk (20 × 7)
instruction ──► hashed words + bigrams ──► EmbeddingBag (mean) ──► MLP ───────────────────────────────────────────────┘
loss: L1 on MEAN_STD-normalized actions, padded steps masked. Executes 10 of 20 actions per chunk.
```

~4 M parameters, fp32, trains on one GPU; the dataloader (PNG decoding) is the bottleneck.
Train: `scripts/shortcut/train/sc-train --preset sc_template_libero_all --run-id tb-tmpl-a`.

## Where to put an idea

| idea                                                                                             | where                                                                                                                     |
| ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| how the instruction is read (pretrained text encoder, token-level attention)                     | `InstructionEncoder`, `hash_tokens` in `modeling_sc_template.py`; a tokenizer step would go in `processor_sc_template.py` |
| where language enters vision                                                                     | `--policy.lang_cond=film\|concat\|none`, `ConvBlock.film`, `ScTemplatePolicy._features`                                   |
| auxiliary losses (instruction-contrastive, counterfactual-instruction penalty, object grounding) | `ScTemplatePolicy.aux_losses` (each returned term is added to the loss and logged)                                        |
| action head (diffusion / flow matching instead of L1 regression)                                 | `ScTemplatePolicy.head`, `forward`, `predict_action_chunk`                                                                |
| new hyper-parameters                                                                             | fields of `ScTemplateConfig` (`--policy.<field>=...` on the sc-train command line)                                        |
| data changes (instructions, suites)                                                              | not here: `sc-train --data-variant` (`scripts/shortcut/train/data_variants.py`)                                           |

`batch["task"]` is the list of instruction strings (already rewritten by the data variant during training);
at evaluation it is the instruction the benchmark gives.
