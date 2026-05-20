---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:37018
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
      value: 0.8608477366255144
      name: Accuracy
    - type: accuracy_threshold
      value: 0.0021666805259883404
      name: Accuracy Threshold
    - type: f1
      value: 0.8620303267257294
      name: F1
    - type: f1_threshold
      value: 0.0007421689806506038
      name: F1 Threshold
    - type: precision
      value: 0.8446736915275123
      name: Precision
    - type: recall
      value: 0.8801152263374485
      name: Recall
    - type: average_precision
      value: 0.9161801478671585
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
    ['Kapan ilmu psikologi dimulai?', 'Abraham Maslow\nPada tahun 1937-1951, Maslow memperdalam ilmunya di Brooklyn College. Di New York, ia bertemu dengan dua mentor lainnya yaitu Ruth Benedict seorang antropologis, dan Max Wertheimer seorang Gestalt psikolog, yang ia kagumi secara profesional maupun personal. Kedua orang inilah yang kemudian menjadi perhatian Maslow dalam mendalami perilaku manusia, kesehatan mental, dan potensi manusia. Ia menulis dalam subjek-subjek ini dengan mendalam. Tulisannya banyak meminjam dari gagasan-gagasan psikologi, namun dengan pengembangan yang signifikan. Penambahan tersebut khususnya mencakup hierarki kebutuhan, berbagai macam kebutuhan, aktualisasi diri seseorang, dan puncak dari pengalaman. Maslow menjadi pelopor aliran humanistik psikologi yang terbentuk pada sekitar tahun 1950 hingga 1960-an. Pada masa ini, ia dikenal sebagai "kekuatan ke tiga" di samping teori Freud dan behaviorisme.'],
    ['Apa ibukota Malaysia?', 'Kuala Lipis\nKuala Lipis merupakan pusat tambang emas sebelum Inggris datang pada 1887. Pada 1898 menjadi ibukota Pahang. Selama masa itu, bangunan kolonial yang megah, seperti Kantor Distrik yang mengesankan dan Clifford School (di mana sultan belajar), dan Pahang Club dibangun. Rumah Residen Inggris yang ada di atas bukit kini merupakan hotel dan museum. Kota ini berkembang dengan datangnya jalur KA pada 1924. Namun, pada tahun 1957, ibukota negeri dipindah ke Kuantan, dan Kuala Lipis jatuh ke dalam kemunduran. Kini merupakan kota yang tidur dan indah dengan pengingat dari sekali masa lampau yang penting.Kuala Lipis juga daerah Parlemen Wakil Perdana Menteri Malaysia sekarang, Datuk Seri Najib Tun Razak dan juga wilayah Parlemen ayah Datuk Najib, Tun Abdul Razak Hussein yang merupakan Perdana Menteri kedua Malaysia.'],
    ['Kapankah zaman prasejarah nerakhir ?', 'Prasejarah\nZaman prasejarah di Indonesia sendiri diperkirakan berakhir pada masa berdirinya Kerajaan Kutai, sekitar abad ke-5; dibuktikan dengan adanya prasasti yang berbentuk yupa yang ditemukan di tepi Sungai Mahakam, Kalimantan Timur baru memasuki era sejarah. Karena tidak terdapat peninggalan catatan tertulis dari zaman prasejarah, keterangan mengenai zaman ini diperoleh melalui bidang-bidang seperti paleontologi, astronomi, biologi, geologi, antropologi, arkeologi. Dalam artian bahwa bukti-bukti prasejarah didapat dari artefak-artefak yang ditemukan di daerah penggalian situs prasejarah.'],
    ['dimanakah letak Sibuhuan?', 'Region Pokémon\nRegion Sinnoh adalah "setting" tempat dalam "Diamond" dan "Pearl", dan terletak di bagian utara Kanto/Johto. Region ini berisi banyak kota, tetapi hanya sedikit rute sungai atau laut. Wilayah ini berisi variasi iklim, di mana ini merupakan pertama kalinya, Pokémon berada di dataran salju. Dengan banyak kota yang padat penduduk, banyak masyarakat yang menggunakan area bawah tanah sebagai markas, selain itu di bagian ini juga banyak fosil yang bisa kita temukan dan dapatkan. Season 10-11 dari serial TV Pokémon mengambil tempat di sini.\nDi Sinnoh ada 23 kota yang berbeda. 8 di antaranya mempunyai Gym, yakni Oreburgh, Eterna, Veilstone, Hearthome, Pastoria, Canalave, Snowpoint, dan Sunyshore.\nAda 14 Pokemon legendaris dari Sinnoh\n-Dialga\n-Palkia\n-Mesprit\n-Uxie\n-Azelf\n-Heatran\n-Regigigas\n-Giratina\n-Cresselia\n-Darkrai\n-Phione\n-Manaphy\n-Shaymin\n-Arceus'],
    ['Kapan Windows 3.1 x pertama dirilis?', 'Microsoft Windows\nSistem operasi Windows telah berevolusi dari MS-DOS, sebuah sistem operasi yang berbasis modus teks dan command-line. Windows versi pertama, Windows Graphic Environment 1.0 pertama kali diperkenalkan pada 10 November 1983, tetapi baru keluar pasar pada bulan November tahun 1985, yang dibuat untuk memenuhi kebutuhan komputer dengan tampilan bergambar. Windows 1.0 merupakan perangkat lunak 16-bit tambahan (bukan merupakan sistem operasi) yang berjalan di atas MS-DOS (dan beberapa varian dari MS-DOS), sehingga ia tidak akan dapat berjalan tanpa adanya sistem operasi DOS. Versi 2.x, versi 3.x juga sama. Beberapa versi terakhir dari Windows (dimulai dari versi 4.0 dan Windows NT 3.1) merupakan sistem operasi mandiri yang tidak lagi bergantung kepada sistem operasi MS-DOS. Microsoft Windows kemudian bisa berkembang dan dapat menguasai penggunaan sistem operasi hingga mencapai 90%.'],
]
scores = model.predict(pairs)
print(scores)
# [3.7170e-05 3.4129e-05 9.9998e-01 3.8087e-05 9.9998e-01]

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'Kapan ilmu psikologi dimulai?',
    [
        'Abraham Maslow\nPada tahun 1937-1951, Maslow memperdalam ilmunya di Brooklyn College. Di New York, ia bertemu dengan dua mentor lainnya yaitu Ruth Benedict seorang antropologis, dan Max Wertheimer seorang Gestalt psikolog, yang ia kagumi secara profesional maupun personal. Kedua orang inilah yang kemudian menjadi perhatian Maslow dalam mendalami perilaku manusia, kesehatan mental, dan potensi manusia. Ia menulis dalam subjek-subjek ini dengan mendalam. Tulisannya banyak meminjam dari gagasan-gagasan psikologi, namun dengan pengembangan yang signifikan. Penambahan tersebut khususnya mencakup hierarki kebutuhan, berbagai macam kebutuhan, aktualisasi diri seseorang, dan puncak dari pengalaman. Maslow menjadi pelopor aliran humanistik psikologi yang terbentuk pada sekitar tahun 1950 hingga 1960-an. Pada masa ini, ia dikenal sebagai "kekuatan ke tiga" di samping teori Freud dan behaviorisme.',
        'Kuala Lipis\nKuala Lipis merupakan pusat tambang emas sebelum Inggris datang pada 1887. Pada 1898 menjadi ibukota Pahang. Selama masa itu, bangunan kolonial yang megah, seperti Kantor Distrik yang mengesankan dan Clifford School (di mana sultan belajar), dan Pahang Club dibangun. Rumah Residen Inggris yang ada di atas bukit kini merupakan hotel dan museum. Kota ini berkembang dengan datangnya jalur KA pada 1924. Namun, pada tahun 1957, ibukota negeri dipindah ke Kuantan, dan Kuala Lipis jatuh ke dalam kemunduran. Kini merupakan kota yang tidur dan indah dengan pengingat dari sekali masa lampau yang penting.Kuala Lipis juga daerah Parlemen Wakil Perdana Menteri Malaysia sekarang, Datuk Seri Najib Tun Razak dan juga wilayah Parlemen ayah Datuk Najib, Tun Abdul Razak Hussein yang merupakan Perdana Menteri kedua Malaysia.',
        'Prasejarah\nZaman prasejarah di Indonesia sendiri diperkirakan berakhir pada masa berdirinya Kerajaan Kutai, sekitar abad ke-5; dibuktikan dengan adanya prasasti yang berbentuk yupa yang ditemukan di tepi Sungai Mahakam, Kalimantan Timur baru memasuki era sejarah. Karena tidak terdapat peninggalan catatan tertulis dari zaman prasejarah, keterangan mengenai zaman ini diperoleh melalui bidang-bidang seperti paleontologi, astronomi, biologi, geologi, antropologi, arkeologi. Dalam artian bahwa bukti-bukti prasejarah didapat dari artefak-artefak yang ditemukan di daerah penggalian situs prasejarah.',
        'Region Pokémon\nRegion Sinnoh adalah "setting" tempat dalam "Diamond" dan "Pearl", dan terletak di bagian utara Kanto/Johto. Region ini berisi banyak kota, tetapi hanya sedikit rute sungai atau laut. Wilayah ini berisi variasi iklim, di mana ini merupakan pertama kalinya, Pokémon berada di dataran salju. Dengan banyak kota yang padat penduduk, banyak masyarakat yang menggunakan area bawah tanah sebagai markas, selain itu di bagian ini juga banyak fosil yang bisa kita temukan dan dapatkan. Season 10-11 dari serial TV Pokémon mengambil tempat di sini.\nDi Sinnoh ada 23 kota yang berbeda. 8 di antaranya mempunyai Gym, yakni Oreburgh, Eterna, Veilstone, Hearthome, Pastoria, Canalave, Snowpoint, dan Sunyshore.\nAda 14 Pokemon legendaris dari Sinnoh\n-Dialga\n-Palkia\n-Mesprit\n-Uxie\n-Azelf\n-Heatran\n-Regigigas\n-Giratina\n-Cresselia\n-Darkrai\n-Phione\n-Manaphy\n-Shaymin\n-Arceus',
        'Microsoft Windows\nSistem operasi Windows telah berevolusi dari MS-DOS, sebuah sistem operasi yang berbasis modus teks dan command-line. Windows versi pertama, Windows Graphic Environment 1.0 pertama kali diperkenalkan pada 10 November 1983, tetapi baru keluar pasar pada bulan November tahun 1985, yang dibuat untuk memenuhi kebutuhan komputer dengan tampilan bergambar. Windows 1.0 merupakan perangkat lunak 16-bit tambahan (bukan merupakan sistem operasi) yang berjalan di atas MS-DOS (dan beberapa varian dari MS-DOS), sehingga ia tidak akan dapat berjalan tanpa adanya sistem operasi DOS. Versi 2.x, versi 3.x juga sama. Beberapa versi terakhir dari Windows (dimulai dari versi 4.0 dan Windows NT 3.1) merupakan sistem operasi mandiri yang tidak lagi bergantung kepada sistem operasi MS-DOS. Microsoft Windows kemudian bisa berkembang dan dapat menguasai penggunaan sistem operasi hingga mencapai 90%.',
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
| accuracy              | 0.8608     |
| accuracy_threshold    | 0.0022     |
| f1                    | 0.862      |
| f1_threshold          | 0.0007     |
| precision             | 0.8447     |
| recall                | 0.8801     |
| **average_precision** | **0.9162** |

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

* Size: 37,018 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 100 samples:
  |          | sentence_0                                                                       | sentence_1                                                                           | label                                                          |
  |:---------|:---------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type     | string                                                                           | string                                                                               | float                                                          |
  | modality | text                                                                             | text                                                                                 |                                                                |
  | details  | <ul><li>min: 6 tokens</li><li>mean: 10.1 tokens</li><li>max: 15 tokens</li></ul> | <ul><li>min: 15 tokens</li><li>mean: 137.09 tokens</li><li>max: 382 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.51</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                        | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | label            |
  |:--------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Kapan ilmu psikologi dimulai?</code>        | <code>Abraham Maslow<br>Pada tahun 1937-1951, Maslow memperdalam ilmunya di Brooklyn College. Di New York, ia bertemu dengan dua mentor lainnya yaitu Ruth Benedict seorang antropologis, dan Max Wertheimer seorang Gestalt psikolog, yang ia kagumi secara profesional maupun personal. Kedua orang inilah yang kemudian menjadi perhatian Maslow dalam mendalami perilaku manusia, kesehatan mental, dan potensi manusia. Ia menulis dalam subjek-subjek ini dengan mendalam. Tulisannya banyak meminjam dari gagasan-gagasan psikologi, namun dengan pengembangan yang signifikan. Penambahan tersebut khususnya mencakup hierarki kebutuhan, berbagai macam kebutuhan, aktualisasi diri seseorang, dan puncak dari pengalaman. Maslow menjadi pelopor aliran humanistik psikologi yang terbentuk pada sekitar tahun 1950 hingga 1960-an. Pada masa ini, ia dikenal sebagai "kekuatan ke tiga" di samping teori Freud dan behaviorisme.</code> | <code>0.0</code> |
  | <code>Apa ibukota Malaysia?</code>                | <code>Kuala Lipis<br>Kuala Lipis merupakan pusat tambang emas sebelum Inggris datang pada 1887. Pada 1898 menjadi ibukota Pahang. Selama masa itu, bangunan kolonial yang megah, seperti Kantor Distrik yang mengesankan dan Clifford School (di mana sultan belajar), dan Pahang Club dibangun. Rumah Residen Inggris yang ada di atas bukit kini merupakan hotel dan museum. Kota ini berkembang dengan datangnya jalur KA pada 1924. Namun, pada tahun 1957, ibukota negeri dipindah ke Kuantan, dan Kuala Lipis jatuh ke dalam kemunduran. Kini merupakan kota yang tidur dan indah dengan pengingat dari sekali masa lampau yang penting.Kuala Lipis juga daerah Parlemen Wakil Perdana Menteri Malaysia sekarang, Datuk Seri Najib Tun Razak dan juga wilayah Parlemen ayah Datuk Najib, Tun Abdul Razak Hussein yang merupakan Perdana Menteri kedua Malaysia.</code>                                                                       | <code>0.0</code> |
  | <code>Kapankah zaman prasejarah nerakhir ?</code> | <code>Prasejarah<br>Zaman prasejarah di Indonesia sendiri diperkirakan berakhir pada masa berdirinya Kerajaan Kutai, sekitar abad ke-5; dibuktikan dengan adanya prasasti yang berbentuk yupa yang ditemukan di tepi Sungai Mahakam, Kalimantan Timur baru memasuki era sejarah. Karena tidak terdapat peninggalan catatan tertulis dari zaman prasejarah, keterangan mengenai zaman ini diperoleh melalui bidang-bidang seperti paleontologi, astronomi, biologi, geologi, antropologi, arkeologi. Dalam artian bahwa bukti-bukti prasejarah didapat dari artefak-artefak yang ditemukan di daerah penggalian situs prasejarah.</code>                                                                                                                                                                                                                                                                                                            | <code>1.0</code> |
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
| 0.2161 | 500  | 0.4249        | -                     |
| 0.4322 | 1000 | 0.2685        | -                     |
| 0.6482 | 1500 | 0.2146        | -                     |
| 0.8643 | 2000 | 0.1570        | -                     |
| 1.0    | 2314 | -             | 0.9311                |
| 1.0804 | 2500 | 0.0995        | -                     |
| 1.2965 | 3000 | 0.0658        | -                     |
| 1.5125 | 3500 | 0.0599        | -                     |
| 1.7286 | 4000 | 0.0377        | -                     |
| 1.9447 | 4500 | 0.0374        | -                     |
| 2.0    | 4628 | -             | 0.9297                |
| 2.1608 | 5000 | 0.0214        | -                     |
| 2.3768 | 5500 | 0.0155        | -                     |
| 2.5929 | 6000 | 0.0124        | -                     |
| 2.8090 | 6500 | 0.0080        | -                     |
| 3.0    | 6942 | -             | 0.9162                |


### Training Time
- **Training**: 51.5 minutes

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