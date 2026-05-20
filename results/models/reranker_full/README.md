---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:121500
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
      value: 0.998880658436214
      name: Accuracy
    - type: accuracy_threshold
      value: 0.12650460004806519
      name: Accuracy Threshold
    - type: f1
      value: 0.9988810820594674
      name: F1
    - type: f1_threshold
      value: 0.12650460004806519
      name: F1 Threshold
    - type: precision
      value: 0.9985031909994079
      name: Precision
    - type: recall
      value: 0.9992592592592593
      name: Recall
    - type: average_precision
      value: 0.999960215723066
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
    ['Kapan Stanisława Walasiewicz lahir?', 'Stanisława Walasiewicz\nWalasiewicz sudah menjadi atlet ternama pada akhir tahun 1920-an. Sebagai atlet amatir, ia mencari nafkah dengan bekerja sebagai klerk di Cleveland. Walaupun bukan warganegara AS, ia tetap bertanding dalam berbagai kejuaraan atletik di Amerika Serikat dengan menggunakan nama Stella Walsh. Ia mendapat hadiah mobil setelah menjuarai perlombaan atletik antarnegara bagian di Cleveland. Walasiewicz ditawari kewarganegaraan AS atas rekomendasi Amateur Athletic Union yang anggotanya ingin Walasiewicz memenangi medali emas untuk AS di Olimpiade Los Angeles 1932. Namun, dua hari sebelum ia mengangkat sumpah sebagai warganegara AS, Walasiewicz berubah pikiran dan mengambil kewarganegaraan Polandia yang ditawarkan Konsulat Polandia di New York City. Pada tahun 1930, pembaca harian "Przegląd Sportowy" memilihnya sebagai atlet Polandia terpopuler tahun 1930.'],
    ['Siapakah vokalis band Arwana?', 'Panbers\nBeberapa tahun kemudian pada tahun 2000-an mereka merekrut musikus biola (violis) berdarah Melayu Kalimantan Hendri Lamiri sebagai membernya sehingga formasi group menjadi berenam. Hendri adalah mantan personel band Arwana (grup musik), sebuah kelompok band asal Pontianak Kalimantan Barat. Benny Panjaitan sang icon Panbers berkeinginan untuk menjadikan musik Panbers selalu bervariasi. Dalam hal ini Musik Panbers kalau ditonton di panggung berbeda dengan plat (piringan hitam) atau CD. Harus lebih bagus yang di panggung agar mereka bisa langgeng. Salah satunya ada nuansa iringan musik biola yang mempercantik alunan lagu mereka. Namun Hendri Lamiri tak selalu bersama Panbers, di luar ia masih kerap membantu penyanyi atau musisi lainnya.'],
    ['apakah palem memiliki buah?', 'Kakapo\nParuh Kakapo secara khusus beradaptasi untuk makan makanan bulat dengan baik. Untuk alasan ini, Kakapo memiliki tembolok burung sangat kecil dibandingkan dengan burung lain seukuran mereka. Mereka merupakan Herbivora umum, memakan tanaman, benih, buah, serbuk sari dan setiap lapisan kayu dalam pohon. Penelitian tahun 1984 mengidentifikasi 25 spesies tanaman sebagai makanan Kakapo. Mereka terutama sekali gemar makan buah tanaman rimu, dan akan memakan buah itu sepanjang musim ketika buahnya melimpah. Kakapo memiliki kebiasaan lain berebut daun atau daun palem dengan makanan dan kulit bagian nutrisi tanaman yang dikeluarkan paruhnya, yang menyisakan gulungan serat yang sulit dicerna. Perdu serat tanaman kecil ini merupakan tanda perbedaan keberadaan Kakapo. Kakapos dipercaya mempekerjakan bakteri dalam foregut untuk memfermentasikan dan membantu masalah pencernaan tanaman.'],
    ['Kapan Singapura merdeka?', 'Lim Yew Hock\nDengan tindakannya kerasnya terhadap huru-hara itu, Inggris lebih percaya diri bila pemerintah setempat memegang keamanan dalam negeri. Lim memimpin delegasi semua partai untuk berunding dengan Britania dalam serangkaian "pembicaraan Merdeka" pada 1956 sampai 1958, dan dengan berhasil membuat Singapura memenangkan konstitusi baru yang memberi kekuasaan sendiri.'],
    ['tahun berapakah Ip Man dilahirkan?', 'Ip Man\nIp Man lahir di Foshan, provinsi Guandong pada tahun 1893 masa pemerintahan Kaisar Guangxu, Dinasti Qing. Dia adalah anak ketiga dari empat bersaudara yang dilahirkan dari pasangan Ip Oi dan Ng Shui. Abang dan kakaknya bernama Ip Kai Gak dan Ip Wan Hum. Ip Man tumbuh dalam keluarga kaya dan menerima pendidikan dengan standar tinggi.'],
]
scores = model.predict(pairs)
print(scores)
# [3.9867e-05 9.9995e-01 9.9995e-01 3.9414e-05 9.9995e-01]

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'Kapan Stanisława Walasiewicz lahir?',
    [
        'Stanisława Walasiewicz\nWalasiewicz sudah menjadi atlet ternama pada akhir tahun 1920-an. Sebagai atlet amatir, ia mencari nafkah dengan bekerja sebagai klerk di Cleveland. Walaupun bukan warganegara AS, ia tetap bertanding dalam berbagai kejuaraan atletik di Amerika Serikat dengan menggunakan nama Stella Walsh. Ia mendapat hadiah mobil setelah menjuarai perlombaan atletik antarnegara bagian di Cleveland. Walasiewicz ditawari kewarganegaraan AS atas rekomendasi Amateur Athletic Union yang anggotanya ingin Walasiewicz memenangi medali emas untuk AS di Olimpiade Los Angeles 1932. Namun, dua hari sebelum ia mengangkat sumpah sebagai warganegara AS, Walasiewicz berubah pikiran dan mengambil kewarganegaraan Polandia yang ditawarkan Konsulat Polandia di New York City. Pada tahun 1930, pembaca harian "Przegląd Sportowy" memilihnya sebagai atlet Polandia terpopuler tahun 1930.',
        'Panbers\nBeberapa tahun kemudian pada tahun 2000-an mereka merekrut musikus biola (violis) berdarah Melayu Kalimantan Hendri Lamiri sebagai membernya sehingga formasi group menjadi berenam. Hendri adalah mantan personel band Arwana (grup musik), sebuah kelompok band asal Pontianak Kalimantan Barat. Benny Panjaitan sang icon Panbers berkeinginan untuk menjadikan musik Panbers selalu bervariasi. Dalam hal ini Musik Panbers kalau ditonton di panggung berbeda dengan plat (piringan hitam) atau CD. Harus lebih bagus yang di panggung agar mereka bisa langgeng. Salah satunya ada nuansa iringan musik biola yang mempercantik alunan lagu mereka. Namun Hendri Lamiri tak selalu bersama Panbers, di luar ia masih kerap membantu penyanyi atau musisi lainnya.',
        'Kakapo\nParuh Kakapo secara khusus beradaptasi untuk makan makanan bulat dengan baik. Untuk alasan ini, Kakapo memiliki tembolok burung sangat kecil dibandingkan dengan burung lain seukuran mereka. Mereka merupakan Herbivora umum, memakan tanaman, benih, buah, serbuk sari dan setiap lapisan kayu dalam pohon. Penelitian tahun 1984 mengidentifikasi 25 spesies tanaman sebagai makanan Kakapo. Mereka terutama sekali gemar makan buah tanaman rimu, dan akan memakan buah itu sepanjang musim ketika buahnya melimpah. Kakapo memiliki kebiasaan lain berebut daun atau daun palem dengan makanan dan kulit bagian nutrisi tanaman yang dikeluarkan paruhnya, yang menyisakan gulungan serat yang sulit dicerna. Perdu serat tanaman kecil ini merupakan tanda perbedaan keberadaan Kakapo. Kakapos dipercaya mempekerjakan bakteri dalam foregut untuk memfermentasikan dan membantu masalah pencernaan tanaman.',
        'Lim Yew Hock\nDengan tindakannya kerasnya terhadap huru-hara itu, Inggris lebih percaya diri bila pemerintah setempat memegang keamanan dalam negeri. Lim memimpin delegasi semua partai untuk berunding dengan Britania dalam serangkaian "pembicaraan Merdeka" pada 1956 sampai 1958, dan dengan berhasil membuat Singapura memenangkan konstitusi baru yang memberi kekuasaan sendiri.',
        'Ip Man\nIp Man lahir di Foshan, provinsi Guandong pada tahun 1893 masa pemerintahan Kaisar Guangxu, Dinasti Qing. Dia adalah anak ketiga dari empat bersaudara yang dilahirkan dari pasangan Ip Oi dan Ng Shui. Abang dan kakaknya bernama Ip Kai Gak dan Ip Wan Hum. Ip Man tumbuh dalam keluarga kaya dan menerima pendidikan dengan standar tinggi.',
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

| Metric                | Value   |
|:----------------------|:--------|
| accuracy              | 0.9989  |
| accuracy_threshold    | 0.1265  |
| f1                    | 0.9989  |
| f1_threshold          | 0.1265  |
| precision             | 0.9985  |
| recall                | 0.9993  |
| **average_precision** | **1.0** |

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

* Size: 121,500 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 100 samples:
  |          | sentence_0                                                                       | sentence_1                                                                          | label                                                          |
  |:---------|:---------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type     | string                                                                           | string                                                                              | float                                                          |
  | modality | text                                                                             | text                                                                                |                                                                |
  | details  | <ul><li>min: 6 tokens</li><li>mean: 9.98 tokens</li><li>max: 18 tokens</li></ul> | <ul><li>min: 20 tokens</li><li>mean: 141.7 tokens</li><li>max: 407 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.58</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                       | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | label            |
  |:-------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Kapan Stanisława Walasiewicz lahir?</code> | <code>Stanisława Walasiewicz<br>Walasiewicz sudah menjadi atlet ternama pada akhir tahun 1920-an. Sebagai atlet amatir, ia mencari nafkah dengan bekerja sebagai klerk di Cleveland. Walaupun bukan warganegara AS, ia tetap bertanding dalam berbagai kejuaraan atletik di Amerika Serikat dengan menggunakan nama Stella Walsh. Ia mendapat hadiah mobil setelah menjuarai perlombaan atletik antarnegara bagian di Cleveland. Walasiewicz ditawari kewarganegaraan AS atas rekomendasi Amateur Athletic Union yang anggotanya ingin Walasiewicz memenangi medali emas untuk AS di Olimpiade Los Angeles 1932. Namun, dua hari sebelum ia mengangkat sumpah sebagai warganegara AS, Walasiewicz berubah pikiran dan mengambil kewarganegaraan Polandia yang ditawarkan Konsulat Polandia di New York City. Pada tahun 1930, pembaca harian "Przegląd Sportowy" memilihnya sebagai atlet Polandia terpopuler tahun 1930.</code>           | <code>0.0</code> |
  | <code>Siapakah vokalis band Arwana?</code>       | <code>Panbers<br>Beberapa tahun kemudian pada tahun 2000-an mereka merekrut musikus biola (violis) berdarah Melayu Kalimantan Hendri Lamiri sebagai membernya sehingga formasi group menjadi berenam. Hendri adalah mantan personel band Arwana (grup musik), sebuah kelompok band asal Pontianak Kalimantan Barat. Benny Panjaitan sang icon Panbers berkeinginan untuk menjadikan musik Panbers selalu bervariasi. Dalam hal ini Musik Panbers kalau ditonton di panggung berbeda dengan plat (piringan hitam) atau CD. Harus lebih bagus yang di panggung agar mereka bisa langgeng. Salah satunya ada nuansa iringan musik biola yang mempercantik alunan lagu mereka. Namun Hendri Lamiri tak selalu bersama Panbers, di luar ia masih kerap membantu penyanyi atau musisi lainnya.</code>                                                                                                                                            | <code>1.0</code> |
  | <code>apakah palem memiliki buah?</code>         | <code>Kakapo<br>Paruh Kakapo secara khusus beradaptasi untuk makan makanan bulat dengan baik. Untuk alasan ini, Kakapo memiliki tembolok burung sangat kecil dibandingkan dengan burung lain seukuran mereka. Mereka merupakan Herbivora umum, memakan tanaman, benih, buah, serbuk sari dan setiap lapisan kayu dalam pohon. Penelitian tahun 1984 mengidentifikasi 25 spesies tanaman sebagai makanan Kakapo. Mereka terutama sekali gemar makan buah tanaman rimu, dan akan memakan buah itu sepanjang musim ketika buahnya melimpah. Kakapo memiliki kebiasaan lain berebut daun atau daun palem dengan makanan dan kulit bagian nutrisi tanaman yang dikeluarkan paruhnya, yang menyisakan gulungan serat yang sulit dicerna. Perdu serat tanaman kecil ini merupakan tanda perbedaan keberadaan Kakapo. Kakapos dipercaya mempekerjakan bakteri dalam foregut untuk memfermentasikan dan membantu masalah pencernaan tanaman.</code> | <code>1.0</code> |
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
| Epoch  | Step  | Training Loss | val_average_precision |
|:------:|:-----:|:-------------:|:---------------------:|
| 0.0658 | 500   | 0.4643        | -                     |
| 0.1317 | 1000  | 0.3923        | -                     |
| 0.1975 | 1500  | 0.3515        | -                     |
| 0.2634 | 2000  | 0.3092        | -                     |
| 0.3292 | 2500  | 0.2725        | -                     |
| 0.3950 | 3000  | 0.2493        | -                     |
| 0.4609 | 3500  | 0.2312        | -                     |
| 0.5267 | 4000  | 0.2085        | -                     |
| 0.5926 | 4500  | 0.2161        | -                     |
| 0.6584 | 5000  | 0.1861        | -                     |
| 0.7243 | 5500  | 0.1883        | -                     |
| 0.7901 | 6000  | 0.1856        | -                     |
| 0.8559 | 6500  | 0.1727        | -                     |
| 0.9218 | 7000  | 0.1685        | -                     |
| 0.9876 | 7500  | 0.1476        | -                     |
| 1.0    | 7594  | -             | 0.9975                |
| 1.0535 | 8000  | 0.1068        | -                     |
| 1.1193 | 8500  | 0.1064        | -                     |
| 1.1851 | 9000  | 0.0975        | -                     |
| 1.2510 | 9500  | 0.1164        | -                     |
| 1.3168 | 10000 | 0.1035        | -                     |
| 1.3827 | 10500 | 0.0954        | -                     |
| 1.4485 | 11000 | 0.0879        | -                     |
| 1.5144 | 11500 | 0.0834        | -                     |
| 1.5802 | 12000 | 0.0690        | -                     |
| 1.6460 | 12500 | 0.0703        | -                     |
| 1.7119 | 13000 | 0.0782        | -                     |
| 1.7777 | 13500 | 0.0732        | -                     |
| 1.8436 | 14000 | 0.0480        | -                     |
| 1.9094 | 14500 | 0.0490        | -                     |
| 1.9752 | 15000 | 0.0536        | -                     |
| 2.0    | 15188 | -             | 0.9996                |
| 2.0411 | 15500 | 0.0374        | -                     |
| 2.1069 | 16000 | 0.0266        | -                     |
| 2.1728 | 16500 | 0.0360        | -                     |
| 2.2386 | 17000 | 0.0308        | -                     |
| 2.3045 | 17500 | 0.0250        | -                     |
| 2.3703 | 18000 | 0.0253        | -                     |
| 2.4361 | 18500 | 0.0330        | -                     |
| 2.5020 | 19000 | 0.0224        | -                     |
| 2.5678 | 19500 | 0.0272        | -                     |
| 2.6337 | 20000 | 0.0171        | -                     |
| 2.6995 | 20500 | 0.0162        | -                     |
| 2.7653 | 21000 | 0.0205        | -                     |
| 2.8312 | 21500 | 0.0144        | -                     |
| 2.8970 | 22000 | 0.0145        | -                     |
| 2.9629 | 22500 | 0.0124        | -                     |
| 3.0    | 22782 | -             | 1.0000                |


### Training Time
- **Training**: 2.1 hours

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