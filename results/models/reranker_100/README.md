---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:3874
- loss:BinaryCrossEntropyLoss
base_model: BAAI/bge-reranker-v2-m3
pipeline_tag: text-ranking
library_name: sentence-transformers
metrics:
- accuracy
- accuracy_threshold
- f1
- f1_threshold
- precision
- recall
- average_precision
model-index:
- name: CrossEncoder based on BAAI/bge-reranker-v2-m3
  results:
  - task:
      type: cross-encoder-binary-classification
      name: Cross Encoder Binary Classification
    dataset:
      name: val
      type: val
    metrics:
    - type: accuracy
      value: 0.7736954732510288
      name: Accuracy
    - type: accuracy_threshold
      value: 0.012165319174528122
      name: Accuracy Threshold
    - type: f1
      value: 0.7830956203417171
      name: F1
    - type: f1_threshold
      value: 0.0003761971602216363
      name: F1 Threshold
    - type: precision
      value: 0.7403903969964949
      name: Precision
    - type: recall
      value: 0.8310288065843622
      name: Recall
    - type: average_precision
      value: 0.8649591850136691
      name: Average Precision
---

# CrossEncoder based on BAAI/bge-reranker-v2-m3

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) <!-- at revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e -->
- **Maximum Sequence Length:** 512 tokens
- **Number of Output Labels:** 1 label
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

### Full Model Architecture

```
CrossEncoder(
  (0): Transformer({'transformer_task': 'sequence-classification', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'logits'}}, 'module_output_name': 'scores', 'architecture': 'XLMRobertaForSequenceClassification'})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of inputs
pairs = [
    ['Berapa tahun sistem wajib sekolah di Indonesia ?', 'Sekolah menengah pertama\nPelajar sekolah menengah pertama umumnya berusia 13-15 tahun. Di Indonesia, setiap warga negara berusia 7-15 tahun tahun wajib mengikuti pendidikan dasar, yakni sekolah dasar (atau sederajat) 6 tahun dan sekolah menengah pertama (atau sederajat) 3 tahun.'],
    ['siapakah ratu yang menjabat paling lama di Imperium Britania?', 'Victoria dari Britania Raya\nPemerintahannya berlangsung 63 tahun, 216 hari, lebih lama dari raja atau ratu Britania Raya manapun sampai 9 September 2015 yang kemudian digantikan oleh anak sulung cicitnya, Ratu Elizabeth II sebagai penguasa Britania Raya terlama. Victoria berhasil mempertahankan keberadaan sistem monarki di Inggris dan menjadikannya sebagai institusi politik seremonial. Pada masa pemerintahan itula, aksi represi terhadap rakyat di kawasan-kawasan koloni Inggris meningkat secara signifikan.'],
    ['apakah lagu pertama yang dirilis GoodBoyBadminton?', 'GoodBoyBadminton\nAlbum Going Out rilis pada tanggal 05 03 2010 dengan single pertama "Kartu Mati" dan menjadi number 1 top chart indie di Ninety Niners pada tanggal 17 04 2010. merilis video klip "Kartu Mati" pada tanggal 22 04 2010.'],
    ['berapakah luas ibukota Mesir ?', "Tangga nada minor\nSebagai contoh, dalam tangga nada A minor melodis naik, melodi tangga nada minor meningkat: A B C D E F# G# A'"],
    ['Berapa jumlah zat kimia dalam tabel periodik ?', 'Unsur kimia\nSifat unsur kimia seringkali dirangkum menggunakan tabel periodik, yang mengatur secara elegan unsur-unsur berdasarkan kenaikan nomor atom menjadi baris ("periode") yang mendasari perulangan sifat kimia dan fisik kolom ("golongan") secara periodik. Tabel standar saat ini berisi 118 unsur yang telah dikonfirmasi per 10 April 2010.'],
]
scores = model.predict(pairs)
print(scores)
# [9.9999e-01 9.9999e-01 9.9999e-01 1.2226e-05 9.9998e-01]

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'Berapa tahun sistem wajib sekolah di Indonesia ?',
    [
        'Sekolah menengah pertama\nPelajar sekolah menengah pertama umumnya berusia 13-15 tahun. Di Indonesia, setiap warga negara berusia 7-15 tahun tahun wajib mengikuti pendidikan dasar, yakni sekolah dasar (atau sederajat) 6 tahun dan sekolah menengah pertama (atau sederajat) 3 tahun.',
        'Victoria dari Britania Raya\nPemerintahannya berlangsung 63 tahun, 216 hari, lebih lama dari raja atau ratu Britania Raya manapun sampai 9 September 2015 yang kemudian digantikan oleh anak sulung cicitnya, Ratu Elizabeth II sebagai penguasa Britania Raya terlama. Victoria berhasil mempertahankan keberadaan sistem monarki di Inggris dan menjadikannya sebagai institusi politik seremonial. Pada masa pemerintahan itula, aksi represi terhadap rakyat di kawasan-kawasan koloni Inggris meningkat secara signifikan.',
        'GoodBoyBadminton\nAlbum Going Out rilis pada tanggal 05 03 2010 dengan single pertama "Kartu Mati" dan menjadi number 1 top chart indie di Ninety Niners pada tanggal 17 04 2010. merilis video klip "Kartu Mati" pada tanggal 22 04 2010.',
        "Tangga nada minor\nSebagai contoh, dalam tangga nada A minor melodis naik, melodi tangga nada minor meningkat: A B C D E F# G# A'",
        'Unsur kimia\nSifat unsur kimia seringkali dirangkum menggunakan tabel periodik, yang mengatur secara elegan unsur-unsur berdasarkan kenaikan nomor atom menjadi baris ("periode") yang mendasari perulangan sifat kimia dan fisik kolom ("golongan") secara periodik. Tabel standar saat ini berisi 118 unsur yang telah dikonfirmasi per 10 April 2010.',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Cross Encoder Binary Classification

* Dataset: `val`
* Evaluated with [<code>CEBinaryClassificationEvaluator</code>](https://sbert.net/docs/package_reference/cross_encoder/evaluation.html#sentence_transformers.cross_encoder.evaluation.CEBinaryClassificationEvaluator)

| Metric                | Value     |
|:----------------------|:----------|
| accuracy              | 0.7737    |
| accuracy_threshold    | 0.0122    |
| f1                    | 0.7831    |
| f1_threshold          | 0.0004    |
| precision             | 0.7404    |
| recall                | 0.831     |
| **average_precision** | **0.865** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 3,874 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 100 samples:
  |          | sentence_0                                                                        | sentence_1                                                                           | label                                                          |
  |:---------|:----------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type     | string                                                                            | string                                                                               | float                                                          |
  | modality | text                                                                              | text                                                                                 |                                                                |
  | details  | <ul><li>min: 7 tokens</li><li>mean: 10.62 tokens</li><li>max: 26 tokens</li></ul> | <ul><li>min: 15 tokens</li><li>mean: 136.66 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.59</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                 | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | label            |
  |:---------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Berapa tahun sistem wajib sekolah di Indonesia ?</code>              | <code>Sekolah menengah pertama<br>Pelajar sekolah menengah pertama umumnya berusia 13-15 tahun. Di Indonesia, setiap warga negara berusia 7-15 tahun tahun wajib mengikuti pendidikan dasar, yakni sekolah dasar (atau sederajat) 6 tahun dan sekolah menengah pertama (atau sederajat) 3 tahun.</code>                                                                                                                                                                                                                                        | <code>1.0</code> |
  | <code>siapakah ratu yang menjabat paling lama di Imperium Britania?</code> | <code>Victoria dari Britania Raya<br>Pemerintahannya berlangsung 63 tahun, 216 hari, lebih lama dari raja atau ratu Britania Raya manapun sampai 9 September 2015 yang kemudian digantikan oleh anak sulung cicitnya, Ratu Elizabeth II sebagai penguasa Britania Raya terlama. Victoria berhasil mempertahankan keberadaan sistem monarki di Inggris dan menjadikannya sebagai institusi politik seremonial. Pada masa pemerintahan itula, aksi represi terhadap rakyat di kawasan-kawasan koloni Inggris meningkat secara signifikan.</code> | <code>1.0</code> |
  | <code>apakah lagu pertama yang dirilis GoodBoyBadminton?</code>            | <code>GoodBoyBadminton<br>Album Going Out rilis pada tanggal 05 03 2010 dengan single pertama "Kartu Mati" dan menjadi number 1 top chart indie di Ninety Niners pada tanggal 17 04 2010. merilis video klip "Kartu Mati" pada tanggal 22 04 2010.</code>                                                                                                                                                                                                                                                                                      | <code>1.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 3
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 16
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss | val_average_precision |
|:------:|:----:|:-------------:|:---------------------:|
| 1.0    | 243  | -             | 0.8637                |
| 2.0    | 486  | -             | 0.8665                |
| 2.0576 | 500  | 0.1381        | -                     |
| 3.0    | 729  | -             | 0.8650                |


### Training Time
- **Training**: 21.9 minutes

### Framework Versions
- Python: 3.12.13
- Sentence Transformers: 5.5.0
- Transformers: 5.8.1
- PyTorch: 2.11.0+cu130
- Accelerate: 1.13.0
- Datasets: 4.8.5
- Tokenizers: 0.22.2

## Additional Resources

- [Training and Finetuning Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-reranker): the end-to-end guide for training or finetuning Cross Encoder (reranker) models.
- [Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers): use text, image, audio, and video reranker models through the same API.
- [Training and Finetuning Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-multimodal-sentence-transformers): training multimodal Cross Encoders.

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->