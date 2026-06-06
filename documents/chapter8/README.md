# Практикум по большим моделям: мультимодальные большие языковые модели

Введение: этот раздел знакомит с распространёнными архитектурами мультимодальных больших языковых моделей и методами их построения

> Появление больших языковых моделей показало всем, что высокоуровневый интеллект в полной мере проявляется в языковой модальности. Как мультимодальные большие языковые модели, способные более полно моделировать реальный мир, достигают более мощного мультимодального понимания и генерации? Могут ли мультимодальные большие языковые модели помочь в достижении AGI?


## Цели этого руководства

1. Познакомиться с типами мультимодальных больших языковых моделей
2. Освоить общий технический фреймворк мультимодальных больших языковых моделей
3. Научиться строить, обучать и запускать инференс мультимодальных больших языковых моделей



## 1. Теоретическая подготовка



### 1.1 Знакомство с типами мультимодальных больших языковых моделей

- Классификация существующих мультимодальных больших языковых моделей по функциональности и поддерживаемым модальностям


> Перед построением мультимодальной большой языковой модели (MLLM) исследователи этой области пришли к общему консенсусу и ключевой предпосылке: благодаря законам масштабирования (scaling laws) и эмерджентным феноменам современные языковые LLM уже обладают мощной способностью к семантическому пониманию, а значит язык стал ключевой модальностью, несущей интеллект, и поэтому языковой интеллект считается узловым центром мультимодального интеллекта. Поэтому почти все MLLM строятся поверх языковых LLM, где LLM выступает в роли центрального модуля принятия решений, подобно мозгу или центральному процессору. Иными словами, добавляя дополнительные внешние нетекстовые модальные модули или энкодеры, LLM наделяется способностями мультимодального восприятия/действия.

Мы разделяем существующие MLLM на разные типы в зависимости от поддерживаемых модальностей и функциональности.

![mllm](assets/MLLM-summary.png)


- Полное руководство: [[Слайды](https://github.com/Lordog/dive-into-llms/blob/main/documents/chapter6/mllms.pdf)] 


- Дополнительные обзоры
    - [A Survey on Multimodal Large Language Models, https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models, 2023](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)
    - [MM-LLMs: Recent Advances in MultiModal Large Language Models, 2023](https://arxiv.org/pdf/2401.13601)


### 1.2 Знакомство с общим техническим фреймворком мультимодальных больших языковых моделей


- Архитектура 1: LLM as Task Scheduler (LLM как планировщик задач)

> В сообществе сейчас существуют две распространённые архитектуры MLLM.
Первая — архитектура «LLM как дискретный планировщик/контроллер». Как показано на рисунке ниже, роль LLM состоит в том, чтобы принимать текстовые сигналы и отдавать текстовые команды нижестоящим модулям. Вся передача сообщений внутри системы происходит через чисто текстовые команды, выводимые LLM, как посредник. Взаимодействия между разными функциональными модулями нет.


![Architecture1](assets/Architecture1.png)

- Архитектура 2: LLM as Joint Part of System (LLM как неотъемлемая часть системы)

> Вторая архитектура — фреймворк «энкодер–LLM–декодер».
Это также сейчас наиболее популярная архитектура. Здесь роль LLM — воспринимать мультимодальную информацию и самостоятельно формировать ответы и действия в структуре «энкодер–LLM–декодер». Поэтому ключевое отличие этой архитектуры от первой в том, что LLM выступает ключевой неотъемлемой частью системы, напрямую получает мультимодальную информацию извне и более плавно делегирует инструкции декодеру/генератору. Во фреймворке «энкодер–LLM–декодер», как показано на рисунке ниже, энкодер обрабатывает закодированные сигналы из нескольких модальностей, LLM выступает центральным лицом, принимающим решения, а декодер управляет мультимодальным выводом.

![Architecture2](assets/Architecture2.png)





## 2. Практика с универсальной мультимодальной большой языковой моделью

> Практикуем процесс построения универсальной мультимодальной большой языковой модели «из любой модальности в любую модальность»


### 2.1 Универсальная унифицированная мультимодальная большая языковая модель «любое-в-любое»: NExT-GPT

> Будущие исследования MLLM непременно будут двигаться в сторону всё более универсальных моделей-универсалов (generalist), а значит будут охватывать как можно больше модальностей и функций. NExT-GPT — одна из наиболее новаторских работ в этой области на сегодняшний день; в ней впервые введена концепция MLLM «любое-в-любое». Эта архитектура реализует мощную функциональность и закладывает фундамент для будущих направлений исследований мультимодальных больших языковых моделей.

![NExT-GPT](assets/NExT-GPT-screen.png)


> Практическая часть этого курса по коду мультимодальных больших языковых моделей будет построена вокруг кода NExT-GPT, который мы разберём и опробуем доступно и по существу.

[NExT-GPT Project](https://next-gpt.github.io/)

[Репозиторий NExT-GPT на GitHub](https://github.com/NExT-GPT/NExT-GPT)




### 2.2 Обзор структуры кода


```
├── figures
├── data
│   ├── T-X_pair_data  
│   │   ├── audiocap                      # text-autio pairs data
│   │   │   ├── audios                    # audio files
│   │   │   └── audiocap.json             # the audio captions
│   │   ├── cc3m                          # text-image paris data
│   │   │   ├── images                    # image files
│   │   │   └── cc3m.json                 # the image captions
│   │   └── webvid                        # text-video pairs data
│   │   │   ├── videos                    # video files
│   │   │   └── webvid.json               # the video captions
│   ├── IT_data                           # instruction data
│   │   ├── T+X-T_data                    # text+[image/audio/video] to text instruction data
│   │   │   ├── alpaca                    # textual instruction data
│   │   │   ├── llava                     # visual instruction data
│   │   ├── T-T+X                         # synthesized text to text+[image/audio/video] instruction data
│   │   └── MosIT                         # Modality-switching Instruction Tuning instruction data
├── code
│   ├── config
│   │   ├── base.yaml                     # the model configuration 
│   │   ├── stage_1.yaml                  # enc-side alignment training configuration
│   │   ├── stage_2.yaml                  # dec-side alignment training configuration
│   │   └── stage_3.yaml                  # instruction-tuning configuration
│   ├── dsconfig
│   │   ├── stage_1.json                  # deepspeed configuration for enc-side alignment training
│   │   ├── stage_2.json                  # deepspeed configuration for dec-side alignment training
│   │   └── stage_3.json                  # deepspeed configuration for instruction-tuning training
│   ├── datast
│   │   ├── base_dataset.py
│   │   ├── catalog.py                    # the catalog information of the dataset
│   │   ├── cc3m_datast.py                # process and load text-image pair dataset
│   │   ├── audiocap_datast.py            # process and load text-audio pair dataset
│   │   ├── webvid_dataset.py             # process and load text-video pair dataset
│   │   ├── T+X-T_instruction_dataset.py  # process and load text+x-to-text instruction dataset
│   │   ├── T-T+X_instruction_dataset.py  # process and load text-to-text+x instruction dataset
│   │   └── concat_dataset.py             # process and load multiple dataset
│   ├── model                     
│   │   ├── ImageBind                     # the code from ImageBind Model
│   │   ├── common
│   │   ├── anyToImageVideoAudio.py       # the main model file
│   │   ├── agent.py
│   │   ├── modeling_llama.py
│   │   ├── custom_ad.py                  # the audio diffusion 
│   │   ├── custom_sd.py                  # the image diffusion
│   │   ├── custom_vd.py                  # the video diffusion
│   │   ├── layers.py                     # the output projection layers
│   │   └── ...  
│   ├── scripts
│   │   ├── train.sh                      # training NExT-GPT script
│   │   └── app.sh                        # deploying demo script
│   ├── header.py
│   ├── process_embeddings.py             # precompute the captions embeddings
│   ├── train.py                          # training
│   ├── inference.py                      # inference
│   ├── demo_app.py                       # deploy Gradio demonstration 
│   └── ...
├── ckpt                           
│   ├── delta_ckpt                        # tunable NExT-GPT params
│   │   ├── nextgpt         
│   │   │   ├── 7b_tiva_v0                # the directory to save the log file
│   │   │   │   ├── log                   # the logs
│   └── ...       
│   ├── pretrained_ckpt                   # frozen params of pretrained modules
│   │   ├── imagebind_ckpt
│   │   │   ├──huge                       # version
│   │   │   │   └──imagebind_huge.pth
│   │   ├── vicuna_ckpt
│   │   │   ├── 7b_v0                     # version
│   │   │   │   ├── config.json
│   │   │   │   ├── pytorch_model-00001-of-00002.bin
│   │   │   │   ├── tokenizer.model
│   │   │   │   └── ...
├── LICENCE.md
├── README.md
└── requirements.txt
```


### 2.3 Установка окружения

Сначала склонируйте репозиторий и установите необходимое окружение; установку можно выполнить, запустив следующие команды:

```
conda env create -n nextgpt python=3.8

conda activate nextgpt

# CUDA 11.6
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

git clone https://github.com/NExT-GPT/NExT-GPT.git
cd NExT-GPT

pip install -r requirements.txt
```




### 2.4 Знакомство с инференсом системы


#### 2.4.1 Загрузка предобученного чекпойнта модели NExT-GPT

- **Шаг 1**: загрузка `замороженных параметров`. [NExT-GPT](https://github.com/NExT-GPT/NExT-GPT) обучается на основе следующих существующих моделей или модулей; пожалуйста, подготовьте чекпойнты согласно инструкциям ниже.
    - `ImageBind` — это унифицированный энкодер изображений/видео/аудио. Предобученный чекпойнт версии `huge` можно скачать [отсюда](https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth). Затем поместите файл `imagebind_huge.pth` в [[./ckpt/pretrained_ckpt/imagebind_ckpt/huge]](ckpt/pretrained_ckpt/imagebind_ckpt/).
    - `Vicuna`: сначала подготовьте LLaMA по инструкции [[здесь]](ckpt/pretrained_ckpt/prepare_vicuna.md). Затем поместите предобученную модель в [[./ckpt/pretrained_ckpt/vicuna_ckpt/]](ckpt/pretrained_ckpt/vicuna_ckpt/).
    - `Image Diffusion` используется для генерации изображений. NExT-GPT использует [Stable Diffusion](https://huggingface.co/runwayml/stable-diffusion-v1-5) версии `v1-5`. (_скачается автоматически в коде_)
    - `Audio Diffusion` используется для генерации аудиоконтента. NExT-GPT использует [AudioLDM](https://github.com/haoheliu/AudioLDM) версии `l-full`. (_скачается автоматически в коде_)
    - `Video Diffusion` используется для генерации видео. Мы используем [ZeroScope](https://huggingface.co/cerspense/zeroscope_v2_576w) версии `v2_576w`. (_скачается автоматически в коде_)

- **Шаг 2**: загрузка `настраиваемых параметров`.

Поместите систему NExT-GPT в [[./ckpt/delta_ckpt/nextgpt/7b_tiva_v0]](./ckpt/delta_ckpt/nextgpt/7b_tiva_v0). Можно выбрать: 1) использовать параметры, обученные вами самостоятельно, либо 2) скачать предобученный чекпойнт с [Huggingface](https://huggingface.co/ChocoWu/nextgpt_7b_tiva_v0).


#### 2.4.2 Развёртывание Gradio Demo

После загрузки чекпойнтов вы можете запустить демо локально следующим образом:
```angular2html
cd ./code
bash scripts/app.sh
```
Ключевые параметры задаются так:
- `--nextgpt_ckpt_path`: путь к предобученным параметрам NExT-GPT.



#### 2.4.3 Практика на тестовых примерах

Текущая версия поддерживает ввод в любой комбинации четырёх модальностей — текст, изображение, видео, звук — и вывод в комбинации модальностей соответственно задаче.
Также поддерживается многораундовое контекстное взаимодействие.

Пожалуйста, запустите тесты самостоятельно, чтобы оценить результат.

- **Case-1**: вход T+I, выход T+A

![case1](assets/T+I-T+A.png)


- **Case-2**: вход T+V, выход T+A

![case2](assets/T+V-T+A.png)


- **Case-3**: вход T+I, выход T+I+V

![case3](assets/T+I-T+I+V.png)


- **Case-4**: вход T, выход T+I+V+A

![case4](assets/T-T+I+V+A.png)







### 2.5 Процесс обучения системы


#### 2.5.1 Подготовка данных

Пожалуйста, скачайте следующие датасеты для обучения модели:

A) Данные пар T-X
  - Данные ***текст-изображение*** `CC3M`: следуйте инструкции [[here]](./data/T-X_pair_data/cc3m/prepare.md). Затем поместите данные в [[./data/T-X_pair_data/cc3m]](./data/T-X_pair_data/cc3m).
  - Данные ***текст-видео*** `WebVid`: см. [[instruction]](./data/T-X_pair_data/webvid/prepare.md). Файлы следует сохранить в [[./data/T-X_pair_data/webvid]](./data/T-X_pair_data/webvid).
  - Данные ***текст-аудио*** `AudioCap`: см. [[instruction]](./data/T-X_pair_data/audiocap/prepare.md). Сохраните данные в [[./data/T-X_pair_data/audiocap]](./data/T-X_pair_data/audiocap).

B) Данные для instruction-тюнинга
  - T+X-T
    - ***Данные визуальных инструкций*** (из `LLaVA`): скачайте [отсюда](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md) и поместите в [[./data/IT_data/T+X-T_data/llava]](./data/IT_data/T+X-T_data/llava/).
    - ***Данные текстовых инструкций*** (из `Alpaca`): скачайте [отсюда](https://github.com/tatsu-lab/stanford_alpaca) и поместите в [[./data/IT_data/T+X-T_data/alpaca/]](data/IT_data/T+X-T_data/alpaca/).
    - ***Данные видеоинструкций*** (из `VideoChat`): скачайте [отсюда](https://github.com/OpenGVLab/InternVideo/tree/main/Data/instruction_data) и поместите в [[./data/IT_data/T+X-T_data/videochat/]](data/IT_data/T+X-T_data/videochat/).

    Примечание: после скачивания датасетов запустите `preprocess_dataset.py`, чтобы предобработать датасеты и привести их к единому формату.
  - T-X+T (T2M)
    - Датасет инструкций `T-X+T` (T2M) сохраняется в [[./data/IT_data/T-T+X_data]](./data/IT_data/T-T+X_data).
   
  - MosIT
    - Инструкции по скачиванию данных получите [здесь](./data/IT_data/MosIT_data/). В итоге поместите их в [[./data/IT_data/MosIT_data/]](./data/IT_data/MosIT_data/).




#### 2.5.2 Подготовка эмбеддингов

При обучении выравнивания на стороне декодера NExT-GPT мы минимизируем расстояние между представлениями Signal Token и подписей (captions). Чтобы обеспечить высокую эффективность системы и сэкономить время и память, мы предвычисляем текстовые эмбеддинги подписей изображений, аудио и видео с помощью текстовых энкодеров из соответствующих диффузионных моделей.

Перед обучением NExT-GPT запустите следующую команду; сгенерированные файлы `embedding` будут сохранены в [[./data/embed]](./data/embed).
```
cd ./code/
python process_embeddings.py ../data/T-X_pair_data/cc3m/cc3m.json image ../data/embed/ runwayml/stable-diffusion-v1-5
```


Описание аргументов:
- args[1]: путь к файлу подписей;
- args[2]: модальность, может быть `image`, `video` или `audio`;
- args[3]: путь для сохранения файла эмбеддингов;
- args[4]: имя соответствующей предобученной диффузионной модели.




#### 2.5.3 Трёхэтапное обучение


Сначала ознакомьтесь с базовым конфигурационным файлом [[./code/config/base.yaml]](./code/config/base.yaml), чтобы понять базовые системные настройки всего модуля.

Затем запустите следующий скрипт, чтобы начать обучение NExT-GPT:
```
cd ./code
bash scripts/train.sh
```

Команда задаётся так:
```
deepspeed --include localhost:0 --master_addr 127.0.0.1 --master_port 28459 train.py \
    --model nextgpt \
    --stage 1\
    --save_path  ../ckpt/delta_ckpt/nextgpt/7b_tiva_v0/\
    --log_path ../ckpt/delta_ckpt/nextgpt/7b_tiva_v0/log/
```
Ключевые параметры:
- `--include`: `localhost:0` означает номер cuda-устройства GPU `0` в deepspeed.
- `--stage`: этап обучения.
- `--save_path`: каталог для сохранения обученных delta-весов. Этот каталог будет создан автоматически.
- `--log_path`: каталог для сохранения лог-файлов.



Всё обучение NExT-GPT делится на 3 шага:

- **Шаг 1**: мультимодальное выравнивание со стороны энкодера с центром на LLM. На этом этапе обучаются ***входные проекционные слои***, при этом ImageBind, LLM и выходные проекционные слои заморожены.
  
  Просто запустите указанный выше скрипт `train.sh`, установив: `--stage 1`
  
  Также см. конфигурационный файл запуска [[./code/config/stage_1.yaml]](./code/config/stage_1.yaml) и конфигурационный файл deepspeed [[./code/dsconfig/stage_1.yaml]](./code/dsconfig/stage_1.yaml) для более подробной пошаговой настройки.

  Обратите внимание, что используемые на этом шаге датасеты перечислены в `dataset_name_list`, и имена датасетов должны точно совпадать с определениями в [[./code/dataset/catalog.py]](./code/dataset/catalog.py).



- **Шаг 2**: выравнивание следования инструкциям со стороны декодера. На этом этапе обучаются ***выходные проекционные слои***, при этом ImageBind, LLM и входные проекционные слои заморожены.

  Просто запустите указанный выше скрипт `train.sh`, установив: `--stage 2`

  Также см. конфигурационный файл запуска [[./code/config/stage_2.yaml]](./code/config/stage_2.yaml) и конфигурационный файл deepspeed [[./code/dsconfig/stage_2.yaml]](./code/dsconfig/stage_2.yaml) для более подробной пошаговой настройки.



- **Шаг 3**: instruction-тюнинг. На этом этапе на датасете инструкций выполняются следующие настройки: 1) тюнинг ***LLM*** через LoRA, 2) тюнинг ***входных проекционных слоёв*** и 3) тюнинг ***выходных проекционных слоёв***.

  Просто запустите указанный выше скрипт `train.sh`, установив: `--stage 3`

  Также см. конфигурационный файл запуска [[./code/config/stage_3.yaml]](./code/config/stage_3.yaml) и конфигурационный файл deepspeed [[./code/dsconfig/stage_3.yaml]](./code/dsconfig/stage_3.yaml) для более подробной пошаговой настройки.
