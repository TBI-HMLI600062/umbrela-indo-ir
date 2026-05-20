---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:18346
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
      value: 0.8203703703703704
      name: Accuracy
    - type: accuracy_threshold
      value: 0.08432211726903915
      name: Accuracy Threshold
    - type: f1
      value: 0.824186555212687
      name: F1
    - type: f1_threshold
      value: 0.00018494337564334273
      name: F1 Threshold
    - type: precision
      value: 0.7843283027284217
      name: Precision
    - type: recall
      value: 0.8683127572016461
      name: Recall
    - type: average_precision
      value: 0.9050088386038224
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
    ['Apakah ajaran inti anarko-komunis?', 'Anarko-Komunisme\nAnarko-Komunisme juga dikenal dengan sebutan anarkis komunisme, komunis anarkisme, anarkisme-komunis ataupun komunisme libertarian. Namun, walaupun semua anarkis komunis adalah komunis libertarian, tetapi tidak semua komunis libertarian adalah anarkis (menganut paham anarkisme), misalnya dewan komunis. hal yang membedakan anarko-komunisme dari varian lain dari libertarian komunisme adalah bentuk oposisinya terhadap segala bentuk kekuasaan politik, hierarki dan dominasi. Komunisme bisa tumbuh subur dinegara - negara miskin maupun negara berkembang, namun dengan runtuhnya negara-negara komunis yang kuat menyebabkan paham-paham komunis inipun tidak akan bisa berkembang menjadi besar.'],
    ['berapakah luas kekuasaan Dinasti Qing?', 'Ci Manuk\nDaerah Aliran Sungai (DAS) Ci Manuk berada dalam pengelolaan Balai Besar Ci Manuk-Ci Sanggarung dan merupakan satu kesatuan aliran sungai Ci Manuk yang terdiri dari 5 Kabupaten yakni Kabupaten Garut, Kabupaten Sumedang, Kabupaten Majalengka, Kabupaten Indramayu dan Kabupaten Cirebon, dan langsung membelah beberapa kota di antaranya adalah Kota Garut, Jatibarang dan Indramayu. Luas DAS Cimanuk adalah 3.584\xa0km dengan panjang total sungai 337,67 Km. Hulu sungai ini berada di Pegunungan Mandalagiri-Puncakgede, Desa Simpang, Kecamatan Cikajang, Kabupaten Garut. Anak sungai besar yang dimiliki Ci Manuk diantaranya adalah:'],
    ['Apa yang dimaksud dengan gereja Katolik Oriental ?', 'Gereja-Gereja Katolik Timur\nHukum kanon yang dimiliki bersama oleh Gereja-Gereja Katolik Timur telah dikodifikasi dalam "Codex Canonum Ecclesiarum Orientalium" (Hukum Kanon Gereja-Gereja Timur) tahun 1990. Dalam "Curia Romana", "dicasterium" yang bekerja sama dengan Gereja-Gereja Katolik Timur adalah Kongregasi bagi Gereja-gereja Oriental, yang, berdasarkan hukum, mencakup sebagai anggota semua batrik dan uskup agung utama Katolik Timur.'],
    ['Siapa penulis Naruto ?', 'Naruto Uzumaki\nSementara Naruto pulih dari cedera, dia meminta Yamato dan Kakashi untuk membawanya ke Negara Besi, agar dia bisa meminta Raikage, untuk memaafkan Sasuke. Yamato dan Kakashi setuju, tetapi saat mereka berbicara dengan Raikage, dia menolak permintaannya dan memarahi Naruto karena telah berdiri untuk penjahat. Naruto pergi ke sebuah penginapan lokal untuk mempertimbangkan apa yang harus dilakukan selanjutnya, dimana dia berhadapan dengan Tobi, yang meminta Naruto memberitahu bagaimana caranya dia bisa membuat Nagato berubah pikiran. Naruto mengabaikan pertanyaan itu, dia menanyakan tentang rencana Tobi dengan Sasuke. Tobi bercerita tentang Rikudo Sennin, Klan Uchiha, kebenaran tentang tragedi klan Uchiha, dan Sasuke akan membalas dendam. Naruto bersikeras bahwa dia masih bisa menghentikan Sasuke, tetapi Tobi tertawa dan pergi, dia mengatakan bahwa Naruto dan Sasuke ditakdirkan untuk bertarung lagi.'],
    ['Apa yang dimaksud dengan molekul ?', 'Fonologi\nKonsonan adalah fonem yang dihasilkan dengan menggerakkan udara keluar dengan rintangan. Dalam hal ini, yang dimaksud dengan rintangan adalah terhambatnya udara keluar oleh adanya gerakan atau perubahan posisi artikulator. Terdapat pula istilah huruf konsonan, yaitu huruf-huruf yang tidak dapat berdiri tunggal dan membutuhkan keberadaan huruf vokal untuk menghasilkan bunyi. Huruf konsonan tersebut terdiri atas: "b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "q", "r", "s", "t", "v", "w", "x", "y", dan "z". Huruf konsonan sering pula disebut sebagai huruf mati.'],
]
scores = model.predict(pairs)
print(scores)
# [9.9998e-01 1.3132e-05 9.9998e-01 1.2289e-05 1.2434e-05]

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'Apakah ajaran inti anarko-komunis?',
    [
        'Anarko-Komunisme\nAnarko-Komunisme juga dikenal dengan sebutan anarkis komunisme, komunis anarkisme, anarkisme-komunis ataupun komunisme libertarian. Namun, walaupun semua anarkis komunis adalah komunis libertarian, tetapi tidak semua komunis libertarian adalah anarkis (menganut paham anarkisme), misalnya dewan komunis. hal yang membedakan anarko-komunisme dari varian lain dari libertarian komunisme adalah bentuk oposisinya terhadap segala bentuk kekuasaan politik, hierarki dan dominasi. Komunisme bisa tumbuh subur dinegara - negara miskin maupun negara berkembang, namun dengan runtuhnya negara-negara komunis yang kuat menyebabkan paham-paham komunis inipun tidak akan bisa berkembang menjadi besar.',
        'Ci Manuk\nDaerah Aliran Sungai (DAS) Ci Manuk berada dalam pengelolaan Balai Besar Ci Manuk-Ci Sanggarung dan merupakan satu kesatuan aliran sungai Ci Manuk yang terdiri dari 5 Kabupaten yakni Kabupaten Garut, Kabupaten Sumedang, Kabupaten Majalengka, Kabupaten Indramayu dan Kabupaten Cirebon, dan langsung membelah beberapa kota di antaranya adalah Kota Garut, Jatibarang dan Indramayu. Luas DAS Cimanuk adalah 3.584\xa0km dengan panjang total sungai 337,67 Km. Hulu sungai ini berada di Pegunungan Mandalagiri-Puncakgede, Desa Simpang, Kecamatan Cikajang, Kabupaten Garut. Anak sungai besar yang dimiliki Ci Manuk diantaranya adalah:',
        'Gereja-Gereja Katolik Timur\nHukum kanon yang dimiliki bersama oleh Gereja-Gereja Katolik Timur telah dikodifikasi dalam "Codex Canonum Ecclesiarum Orientalium" (Hukum Kanon Gereja-Gereja Timur) tahun 1990. Dalam "Curia Romana", "dicasterium" yang bekerja sama dengan Gereja-Gereja Katolik Timur adalah Kongregasi bagi Gereja-gereja Oriental, yang, berdasarkan hukum, mencakup sebagai anggota semua batrik dan uskup agung utama Katolik Timur.',
        'Naruto Uzumaki\nSementara Naruto pulih dari cedera, dia meminta Yamato dan Kakashi untuk membawanya ke Negara Besi, agar dia bisa meminta Raikage, untuk memaafkan Sasuke. Yamato dan Kakashi setuju, tetapi saat mereka berbicara dengan Raikage, dia menolak permintaannya dan memarahi Naruto karena telah berdiri untuk penjahat. Naruto pergi ke sebuah penginapan lokal untuk mempertimbangkan apa yang harus dilakukan selanjutnya, dimana dia berhadapan dengan Tobi, yang meminta Naruto memberitahu bagaimana caranya dia bisa membuat Nagato berubah pikiran. Naruto mengabaikan pertanyaan itu, dia menanyakan tentang rencana Tobi dengan Sasuke. Tobi bercerita tentang Rikudo Sennin, Klan Uchiha, kebenaran tentang tragedi klan Uchiha, dan Sasuke akan membalas dendam. Naruto bersikeras bahwa dia masih bisa menghentikan Sasuke, tetapi Tobi tertawa dan pergi, dia mengatakan bahwa Naruto dan Sasuke ditakdirkan untuk bertarung lagi.',
        'Fonologi\nKonsonan adalah fonem yang dihasilkan dengan menggerakkan udara keluar dengan rintangan. Dalam hal ini, yang dimaksud dengan rintangan adalah terhambatnya udara keluar oleh adanya gerakan atau perubahan posisi artikulator. Terdapat pula istilah huruf konsonan, yaitu huruf-huruf yang tidak dapat berdiri tunggal dan membutuhkan keberadaan huruf vokal untuk menghasilkan bunyi. Huruf konsonan tersebut terdiri atas: "b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "q", "r", "s", "t", "v", "w", "x", "y", dan "z". Huruf konsonan sering pula disebut sebagai huruf mati.',
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
| accuracy              | 0.8204    |
| accuracy_threshold    | 0.0843    |
| f1                    | 0.8242    |
| f1_threshold          | 0.0002    |
| precision             | 0.7843    |
| recall                | 0.8683    |
| **average_precision** | **0.905** |

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

* Size: 18,346 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 100 samples:
  |          | sentence_0                                                                       | sentence_1                                                                           | label                                                          |
  |:---------|:---------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type     | string                                                                           | string                                                                               | float                                                          |
  | modality | text                                                                             | text                                                                                 |                                                                |
  | details  | <ul><li>min: 6 tokens</li><li>mean: 10.3 tokens</li><li>max: 18 tokens</li></ul> | <ul><li>min: 15 tokens</li><li>mean: 149.44 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.51</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                      | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | label            |
  |:----------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Apakah ajaran inti anarko-komunis?</code>                 | <code>Anarko-Komunisme<br>Anarko-Komunisme juga dikenal dengan sebutan anarkis komunisme, komunis anarkisme, anarkisme-komunis ataupun komunisme libertarian. Namun, walaupun semua anarkis komunis adalah komunis libertarian, tetapi tidak semua komunis libertarian adalah anarkis (menganut paham anarkisme), misalnya dewan komunis. hal yang membedakan anarko-komunisme dari varian lain dari libertarian komunisme adalah bentuk oposisinya terhadap segala bentuk kekuasaan politik, hierarki dan dominasi. Komunisme bisa tumbuh subur dinegara - negara miskin maupun negara berkembang, namun dengan runtuhnya negara-negara komunis yang kuat menyebabkan paham-paham komunis inipun tidak akan bisa berkembang menjadi besar.</code> | <code>1.0</code> |
  | <code>berapakah luas kekuasaan Dinasti Qing?</code>             | <code>Ci Manuk<br>Daerah Aliran Sungai (DAS) Ci Manuk berada dalam pengelolaan Balai Besar Ci Manuk-Ci Sanggarung dan merupakan satu kesatuan aliran sungai Ci Manuk yang terdiri dari 5 Kabupaten yakni Kabupaten Garut, Kabupaten Sumedang, Kabupaten Majalengka, Kabupaten Indramayu dan Kabupaten Cirebon, dan langsung membelah beberapa kota di antaranya adalah Kota Garut, Jatibarang dan Indramayu. Luas DAS Cimanuk adalah 3.584 km dengan panjang total sungai 337,67 Km. Hulu sungai ini berada di Pegunungan Mandalagiri-Puncakgede, Desa Simpang, Kecamatan Cikajang, Kabupaten Garut. Anak sungai besar yang dimiliki Ci Manuk diantaranya adalah:</code>                                                                           | <code>0.0</code> |
  | <code>Apa yang dimaksud dengan gereja Katolik Oriental ?</code> | <code>Gereja-Gereja Katolik Timur<br>Hukum kanon yang dimiliki bersama oleh Gereja-Gereja Katolik Timur telah dikodifikasi dalam "Codex Canonum Ecclesiarum Orientalium" (Hukum Kanon Gereja-Gereja Timur) tahun 1990. Dalam "Curia Romana", "dicasterium" yang bekerja sama dengan Gereja-Gereja Katolik Timur adalah Kongregasi bagi Gereja-gereja Oriental, yang, berdasarkan hukum, mencakup sebagai anggota semua batrik dan uskup agung utama Katolik Timur.</code>                                                                                                                                                                                                                                                                          | <code>1.0</code> |
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
| 0.4359 | 500  | 0.3708        | -                     |
| 0.8718 | 1000 | 0.1706        | -                     |
| 1.0    | 1147 | -             | 0.8959                |
| 1.3078 | 1500 | 0.0666        | -                     |
| 1.7437 | 2000 | 0.0218        | -                     |
| 2.0    | 2294 | -             | 0.9043                |
| 2.1796 | 2500 | 0.0220        | -                     |
| 2.6155 | 3000 | 0.0097        | -                     |
| 3.0    | 3441 | -             | 0.9050                |


### Training Time
- **Training**: 34.7 minutes

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