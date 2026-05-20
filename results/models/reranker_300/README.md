---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:10570
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
      value: 0.7971604938271605
      name: Accuracy
    - type: accuracy_threshold
      value: 0.006200859323143959
      name: Accuracy Threshold
    - type: f1
      value: 0.8006833622059187
      name: F1
    - type: f1_threshold
      value: 0.0005175767000764608
      name: F1 Threshold
    - type: precision
      value: 0.7773317005874785
      name: Precision
    - type: recall
      value: 0.8254814814814815
      name: Recall
    - type: average_precision
      value: 0.8727133441608841
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
    ['Kapan Singapura merdeka?', 'Sejarah Singapura\nSelepas perang, penduduk setempat dibenarkan menjalankan pemerintahan sendiri tetapi masih belum mencapai kemerdekaan. Seterusnya pada tahun 1963 Singapura telah bergabung dengan Tanah Melayu bersama-sama dengan Sabah dan Sarawak untuk membentuk Malaysia. Tetapi Singapura dikeluarkan dari Malaysia dan menjadi sebuah republik pada 9 Agustus 1965.'],
    ['apakah nama kitab suci Kristen?', 'Gereja Yesus Kristus dari Orang-orang Suci Zaman Akhir\nPada awal 1830 naskah itu terwujud secara mujizat dan menjadi sebuah buku dan diberi nama Kitab Mormon. Kitab itu merupakan kitab suci baru yang diterjemahkan dari lempengan emas. Menurut Thomas O’Dea, tema dalam Kitab Mormon adalah tiba dan bermukimnya orang Ibrani di benua Amerika sebelum era kekristenan. Tema ini pas dengan maksud untuk menjelaskan asal usul orang indian di Amerika, yang pada masa Joseph Smith banyak diperdebatkan.'],
    ['Siapakah pencipta anime Naruto?', 'Daftar karakter Naruto\nSaat mengembangkan seri ini, Kishimoto menciptakan tiga karakter utama sebagai dasar untuk desain dari tim lainnya. Dia juga menggunakan karakter di "shōnen" manga lainnya sebagai referensi dalam mendesain karakter, keputusan yang dikritik oleh beberapa penerbit anime dan manga. Akan tetapi karakter yang dikembangkan oleh Kishimoto ini mendapatkan pujian karena menggabungkan banyak aspek yang lebih baik dari karakter "shōnen" sebelumnya. Presentasi visual dari karakter dikomentari oleh beberapa pengulas, dengan pujian dan kritik yang diberikan untuk karya Kishimoto dalam manga dan adaptasi anime.'],
    ['Apa nama mata uang Korea Selatan ?', 'Won\nWon (圓; simbol: ₩) adalah mata uang di Korea Utara dan Korea Selatan. Won dibagi menjadi 100 "chon" (錢; di Korea Selatan juga dieja "jeon"). Won diperkenalkan sebagai mata uang Korea pada tahun 1902 menggantikan yen. Pada tahun 1910, seiring dengan pendudukan Jepang terhadap Korea, won digantikan oleh yen.\nPada tahun 1945, dengan berpisahnya Korea, mata uang pun dipisah menjadi dua, masing-masing tetap disebut won. Kode ISO 4217 won Korea Selatan adalah codice_1 sedangkan won Korea Utara codice_2.\nSebagai perbandingan, berikut adalah nilai tukar kedua mata uang won terhadap dolar AS (codice_3) pada 16 Juli 2013 menurut XE.com:'],
    ['Apakah sumber aliran genre Death metal?', 'Death (band)\nMereka di akui sebagai salah satu grup paling berpengaruh di genre death metal. Debut album band ""Scream Bloody Gore"" dianggap sebagai pola genre tersebut, dijelaskan oleh para kritikus sebagai ""Dokumen prototype pertama death metal"". Hanya Schuldiner salah satu orang yang pertama tersisa di band dari awal hingga akhir. Para penulis biografi musik telah menganugerahi Schuldiner sebagai ""bapak death metal"."'],
]
scores = model.predict(pairs)
print(scores)
# [9.9999e-01 1.0503e-05 1.0984e-05 9.9999e-01 9.9999e-01]

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'Kapan Singapura merdeka?',
    [
        'Sejarah Singapura\nSelepas perang, penduduk setempat dibenarkan menjalankan pemerintahan sendiri tetapi masih belum mencapai kemerdekaan. Seterusnya pada tahun 1963 Singapura telah bergabung dengan Tanah Melayu bersama-sama dengan Sabah dan Sarawak untuk membentuk Malaysia. Tetapi Singapura dikeluarkan dari Malaysia dan menjadi sebuah republik pada 9 Agustus 1965.',
        'Gereja Yesus Kristus dari Orang-orang Suci Zaman Akhir\nPada awal 1830 naskah itu terwujud secara mujizat dan menjadi sebuah buku dan diberi nama Kitab Mormon. Kitab itu merupakan kitab suci baru yang diterjemahkan dari lempengan emas. Menurut Thomas O’Dea, tema dalam Kitab Mormon adalah tiba dan bermukimnya orang Ibrani di benua Amerika sebelum era kekristenan. Tema ini pas dengan maksud untuk menjelaskan asal usul orang indian di Amerika, yang pada masa Joseph Smith banyak diperdebatkan.',
        'Daftar karakter Naruto\nSaat mengembangkan seri ini, Kishimoto menciptakan tiga karakter utama sebagai dasar untuk desain dari tim lainnya. Dia juga menggunakan karakter di "shōnen" manga lainnya sebagai referensi dalam mendesain karakter, keputusan yang dikritik oleh beberapa penerbit anime dan manga. Akan tetapi karakter yang dikembangkan oleh Kishimoto ini mendapatkan pujian karena menggabungkan banyak aspek yang lebih baik dari karakter "shōnen" sebelumnya. Presentasi visual dari karakter dikomentari oleh beberapa pengulas, dengan pujian dan kritik yang diberikan untuk karya Kishimoto dalam manga dan adaptasi anime.',
        'Won\nWon (圓; simbol: ₩) adalah mata uang di Korea Utara dan Korea Selatan. Won dibagi menjadi 100 "chon" (錢; di Korea Selatan juga dieja "jeon"). Won diperkenalkan sebagai mata uang Korea pada tahun 1902 menggantikan yen. Pada tahun 1910, seiring dengan pendudukan Jepang terhadap Korea, won digantikan oleh yen.\nPada tahun 1945, dengan berpisahnya Korea, mata uang pun dipisah menjadi dua, masing-masing tetap disebut won. Kode ISO 4217 won Korea Selatan adalah codice_1 sedangkan won Korea Utara codice_2.\nSebagai perbandingan, berikut adalah nilai tukar kedua mata uang won terhadap dolar AS (codice_3) pada 16 Juli 2013 menurut XE.com:',
        'Death (band)\nMereka di akui sebagai salah satu grup paling berpengaruh di genre death metal. Debut album band ""Scream Bloody Gore"" dianggap sebagai pola genre tersebut, dijelaskan oleh para kritikus sebagai ""Dokumen prototype pertama death metal"". Hanya Schuldiner salah satu orang yang pertama tersisa di band dari awal hingga akhir. Para penulis biografi musik telah menganugerahi Schuldiner sebagai ""bapak death metal"."',
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

| Metric                | Value      |
|:----------------------|:-----------|
| accuracy              | 0.7972     |
| accuracy_threshold    | 0.0062     |
| f1                    | 0.8007     |
| f1_threshold          | 0.0005     |
| precision             | 0.7773     |
| recall                | 0.8255     |
| **average_precision** | **0.8727** |

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

* Size: 10,570 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 100 samples:
  |          | sentence_0                                                                       | sentence_1                                                                           | label                                                         |
  |:---------|:---------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type     | string                                                                           | string                                                                               | float                                                         |
  | modality | text                                                                             | text                                                                                 |                                                               |
  | details  | <ul><li>min: 7 tokens</li><li>mean: 9.67 tokens</li><li>max: 26 tokens</li></ul> | <ul><li>min: 15 tokens</li><li>mean: 165.63 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.5</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                   | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | label            |
  |:---------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Kapan Singapura merdeka?</code>        | <code>Sejarah Singapura<br>Selepas perang, penduduk setempat dibenarkan menjalankan pemerintahan sendiri tetapi masih belum mencapai kemerdekaan. Seterusnya pada tahun 1963 Singapura telah bergabung dengan Tanah Melayu bersama-sama dengan Sabah dan Sarawak untuk membentuk Malaysia. Tetapi Singapura dikeluarkan dari Malaysia dan menjadi sebuah republik pada 9 Agustus 1965.</code>                                                                                                                                                                                                                                                                      | <code>1.0</code> |
  | <code>apakah nama kitab suci Kristen?</code> | <code>Gereja Yesus Kristus dari Orang-orang Suci Zaman Akhir<br>Pada awal 1830 naskah itu terwujud secara mujizat dan menjadi sebuah buku dan diberi nama Kitab Mormon. Kitab itu merupakan kitab suci baru yang diterjemahkan dari lempengan emas. Menurut Thomas O’Dea, tema dalam Kitab Mormon adalah tiba dan bermukimnya orang Ibrani di benua Amerika sebelum era kekristenan. Tema ini pas dengan maksud untuk menjelaskan asal usul orang indian di Amerika, yang pada masa Joseph Smith banyak diperdebatkan.</code>                                                                                                                                      | <code>0.0</code> |
  | <code>Siapakah pencipta anime Naruto?</code> | <code>Daftar karakter Naruto<br>Saat mengembangkan seri ini, Kishimoto menciptakan tiga karakter utama sebagai dasar untuk desain dari tim lainnya. Dia juga menggunakan karakter di "shōnen" manga lainnya sebagai referensi dalam mendesain karakter, keputusan yang dikritik oleh beberapa penerbit anime dan manga. Akan tetapi karakter yang dikembangkan oleh Kishimoto ini mendapatkan pujian karena menggabungkan banyak aspek yang lebih baik dari karakter "shōnen" sebelumnya. Presentasi visual dari karakter dikomentari oleh beberapa pengulas, dengan pujian dan kritik yang diberikan untuk karya Kishimoto dalam manga dan adaptasi anime.</code> | <code>0.0</code> |
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
| 0.7564 | 500  | 0.2943        | -                     |
| 1.0    | 661  | -             | 0.8839                |
| 1.5129 | 1000 | 0.0549        | -                     |
| 2.0    | 1322 | -             | 0.8785                |
| 2.2693 | 1500 | 0.0089        | -                     |
| 3.0    | 1983 | -             | 0.8727                |


### Training Time
- **Training**: 27.9 minutes

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