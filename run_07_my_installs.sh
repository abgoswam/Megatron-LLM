
# ======================
--use_checkpoint_args
influences call to load_args_from_checkpoint in checkpointing.py
Else: arguments not loaded up.

# ======================
sentencepiece
Else error at tokenizer.py (line 34)

# ======================
Install the megatron/data/helpers binary:
cd megatron/data/
make
cd ../../

Else
[rank0]: Traceback (most recent call last):
[rank0]:   File "/tmp/amlt-code-download/abgoswam-Megatron-LLM/finetune.py", line 268, in <module>
[rank0]:     pretrain(args, data_provider, model_provider,  ModelType.encoder_or_decoder,
[rank0]:   File "/tmp/amlt-code-download/abgoswam-Megatron-LLM/megatron/training.py", line 127, in pretrain
[rank0]:     = build_train_valid_test_data_iterators(
[rank0]:   File "/tmp/amlt-code-download/abgoswam-Megatron-LLM/megatron/training.py", line 911, in build_train_valid_test_data_iterators
[rank0]:     train_ds, valid_ds, test_ds = build_train_valid_test_datasets_provider(
[rank0]:   File "/tmp/amlt-code-download/abgoswam-Megatron-LLM/finetune.py", line 179, in data_provider
[rank0]:     train_ds, valid_ds, test_ds = builder(
[rank0]:   File "/tmp/amlt-code-download/abgoswam-Megatron-LLM/megatron/data/gpt_dataset.py", line 35, in build_train_valid_test_datasets
[rank0]:     return _build_train_valid_test_datasets(data_prefix[0],
[rank0]:   File "/tmp/amlt-code-download/abgoswam-Megatron-LLM/megatron/data/gpt_dataset.py", line 199, in _build_train_valid_test_datasets
[rank0]:     train_dataset = _f(0, 'train')
[rank0]:   File "/tmp/amlt-code-download/abgoswam-Megatron-LLM/megatron/data/gpt_dataset.py", line 193, in _f
[rank0]:     dataset = GPTDataset(name, data_prefix,
[rank0]:   File "/tmp/amlt-code-download/abgoswam-Megatron-LLM/megatron/data/gpt_dataset.py", line 234, in __init__
[rank0]:     self.doc_idx, self.sample_idx, self.shuffle_idx = _build_index_mappings(
[rank0]:   File "/tmp/amlt-code-download/abgoswam-Megatron-LLM/megatron/data/gpt_dataset.py", line 354, in _build_index_mappings
[rank0]:     from megatron.data import helpers
[rank0]: ImportError: cannot import name 'helpers' from 'megatron.data' (/tmp/amlt-code-download/abgoswam-Megatron-LLM/megatron/data/__init__.py)