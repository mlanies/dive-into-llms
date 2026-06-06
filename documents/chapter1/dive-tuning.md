# Дообучение и развёртывание предобученных языковых моделей

> Введение: этот раздел знакомит с дообучением (fine-tuning) предобученных моделей
Хотите повысить качество предобученной модели на конкретной задаче? Давайте выберем подходящую предобученную модель, дообучим её под конкретную задачу и развернём дообученную модель в виде удобного демо!
## Цели этого руководства:
1. Освоить работу с инструментарием Transformers
2. Научиться дообучению и инференсу предобученных моделей (разделяемая настраиваемая версия & версия с интеграцией по умолчанию)
3. Научиться развёртывать демо с помощью Gradio Spaces
4. Понять, как выбирать разные типы предобученных моделей и сценарии их применения

## Содержание руководства:

### 1. Подготовка к работе:
#### 1.1 Знакомство с инструментарием: Transformers
https://github.com/huggingface/transformers

> 🤗 Transformers предоставляет API и инструменты, позволяющие легко скачивать и обучать передовые предобученные модели. Использование предобученных моделей снижает вычислительные затраты и выбросы углерода, а также экономит время и ресурсы, необходимые для обучения с нуля. Эти модели поддерживают распространённые задачи в разных модальностях, например:
📝 Обработка естественного языка: классификация текста, распознавание именованных сущностей, ответы на вопросы, языковое моделирование, суммаризация, перевод, выбор из вариантов и генерация текста.
🖼️ Компьютерное зрение: классификация изображений, детекция объектов и семантическая сегментация.
🗣️ Аудио: автоматическое распознавание речи и классификация аудио.
🐙 Мультимодальность: ответы на вопросы по таблицам, оптическое распознавание символов, извлечение информации из отсканированных документов, классификация видео и визуальные ответы на вопросы.

Подробная документация (на китайском): https://huggingface.co/docs/transformers/main/zh/index

![huggingface](./assets/huggingface.PNG)

#### 1.2 Установка окружения: на примере классификации текста (например, детекции фейковых новостей)
1. Заходим в репозиторий примеров по классификации текста, изучаем по readme ключевые параметры, скачиваем requirements.txt и run_classification.py

https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification

2. Установка окружения:
- 1. Создаём новое окружение через conda: conda create -n llm python=3.9
- 2. Активируем виртуальное окружение: conda activate llm
- 3. pip install transformers
- 4. Удаляем из requirements.txt автоматически устанавливаемый torch, затем pip install -r requirements.txt

> Если скорость загрузки низкая, можно использовать китайские зеркала: pip [Packages] -i https://pypi.tuna.tsinghua.edu.cn/simple

> При установке pytorch с китайского зеркала автоматически выберется CPU-версия pytorch, которая не сможет работать на GPU, поэтому ——

- 5. conda install pytorch

> Если скорость загрузки низкая, можно настроить зеркало conda по этому блогу: https://blog.csdn.net/weixin_42797483/article/details/132048218

3. Подготовка данных: возьмём для примера датасет фейковых твитов с Kaggle: https://www.kaggle.com/c/nlp-getting-started/data

#### 1.3 Готовые проектные пакеты (демонстрационный код и данные)
（1）Разделяемая настраиваемая версия (ключевые модули разделены, что упрощает понимание; можно кастомизировать загрузку данных, структуру модели, метрики оценки и т. д.)

[Ссылка для скачивания TextClassificationCustom](https://drive.google.com/file/d/12cVWpYbhKVLTqOEKbeyj_4WcFzLd_KJX/view?usp=drive_link)

（2）Версия с интеграцией по умолчанию (код более насыщенный и сложный, обычно вызывается напрямую через гиперпараметры, есть небольшой порог входа в разработку)

[Ссылка для скачивания TextClassification](https://drive.google.com/file/d/10jnqREVDddmOUH4sbHvl-LiPn6uxj57B/view?usp=drive_link)



### 2. Кастомная разработка на основе разделяемой версии (минимально жизнеспособный продукт, MVP)
Всего три основных файла: main.py — главная программа, utils_data.py — файл загрузки и обработки данных, modeling_bert.py — файл со структурой модели

![project structure](./assets/0.png)

#### 2.1 Разбираем ключевые модули
1. Загрузка и обработка данных (utils_data.py)
![utils_data.py](./assets/1.png)

2. Загрузка модели (modeling_bert.py)

![modeling_bert_1.py](./assets/2.png)

![modeling_bert_2.py](./assets/3.png)


3. Обучение/валидация/предсказание (main.py)

![main.py](./assets/4.png)

#### 2.2 Запуск обучения/валидации/предсказания «под ключ»
```shell
python main.py
```


### 3. Дообучение на основе интегрированной версии (опционально, на базе run_classification.py)

#### 3.1 Разбираем ключевые модули:
1. Загрузка данных (формат csv или json)
![load data](./assets/5.png)

2. Обработка данных
![process data](./assets/6.png)

3. Загрузка модели
![load model](./assets/7.png)

4. Обучение/валидация/предсказание
![train dev predict](./assets/8.png)

#### 3.2 Обучение модели
Одновременно проводим валидацию на dev-наборе и предсказание на тестовом наборе, выполнив следующий скрипт:

```shell
python run_classification.py \
    --model_name_or_path  bert-base-uncased \
    --train_file data/train.csv \
    --validation_file data/val.csv \
    --test_file data/test.csv \
    --shuffle_train_dataset \
    --metric_name accuracy \
    --text_column_name "text" \
    --text_column_delimiter "\n" \
    --label_column_name "target" \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length 512 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --output_dir experiments/
```

Если возникает ошибка или всё зависает, обычно это проблемы с сетью:

1. <u>При скачивании модели показывается «Network is unreachable» — скачайте модель вручную локально: https://huggingface.co/google-bert/bert-base-uncased</u>
2. <u>Если после загрузки данных всё зависает, а после прерывания по CTRL+C видно, что застряло на «connection», — это значит, что при загрузке метрик оценки пакетом evaluate не удалось установить сетевое соединение.</u>

![bug](./assets/9.png)

В этом случае можно скачать весь пакет с GitHub проекта evaluate: https://github.com/huggingface/evaluate/tree/main, и в гиперпараметрах поменять путь --metric_name на локальный путь к метрике, то есть:

```shell
python run_classification.py \
    --model_name_or_path  bert-base-uncased \
    --train_file data/train.csv \
    --validation_file data/val.csv \
    --test_file data/test.csv \
    --shuffle_train_dataset \
    --metric_name evaluate/metrics/accuracy/accuracy.py \
    --text_column_name "text" \
    --text_column_delimiter "\n" \
    --label_column_name "target" \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length 512 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --output_dir experiments/
```

![reference result](./assets/10.png)



### 4. Развёртывание модели: после завершения обучения модель можно развернуть в онлайн-демо на Gradio Spaces
#### 4.1 Руководство по Gradio Spaces
https://huggingface.co/docs/hub/en/spaces-sdks-gradio
#### 4.2 Создание spaces
1. https://huggingface.co/new-space?sdk=gradio
2. Примечание: если страница не открывается, попробуйте воспользоваться VPN

![Gradio Spaces](./assets/gradio.png)

#### 4.3 Ключевой код инференса
Подробности см. в файле app.py из проектного пакета

![app.py](./assets/11.png)

#### 4.4 Загрузка app.py, файла конфигурации окружения и модели в Gradio Spaces
1. Файл конфигурации (requirements.txt)
```
transformers==4.30.2
torch==2.0.0
```

2. Обзор файлов

![file overview](./assets/12.png)


3. Результат демо
Пример успешно развёрнутого демо для справки: https://huggingface.co/spaces/cooelf/text-classification
В правом верхнем углу во вкладке «Files» можно посмотреть исходный код.

![Files](./assets/13.png)


4. Бонус: на платформе Spaces можно увидеть популярные демо недели, а также искать интересующие вас большие модели и демо, чтобы попробовать их
![Spaces](./assets/14.png)



### 5. Продвинутые упражнения
1. Попробуйте другие задачи классификации/регрессии, например классификацию тональности, классификацию новостей, классификацию уязвимостей и т. д.
2. Попробуйте другие типы моделей, например T5, ELECTRA и т. д.


### 6. Другие часто используемые модели
1. Модели для ответов на вопросы: https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering
2. Суммаризация текста: https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization
3. Вызов Llama2 для инференса: https://huggingface.co/docs/transformers/en/model_doc/llama2
4. Лёгкое дообучение Llama2 (LoRA):  https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/5-Fine%20Tuning/LoRA_Tuning_PEFT.ipynb


### 7. Дополнительное чтение
1. Всесторонний и глубокий 43-страничный [обзор](https://mp.weixin.qq.com/s?__biz=Mzk0MTYzMzMxMA==&mid=2247484539&idx=1&sn=6ee42baab4ad792e74ac6a89d7dd87d9&chksm=c2ce3e0af5b9b71c578cb6836e09cce5f60b4a1cfffadc1c4c98211fc0af621bfaaaa8fe0aff&mpshare=1&scene=24&srcid=0331FO6iqVqs5Vx3iHrtqouQ&sharer_shareinfo=45c3fb78d9cfb9627908da44dd7f5559&sharer_shareinfo_first=45c3fb78d9cfb9627908da44dd7f5559&key=a2847c972f830c4143e00e0430f657d8ab5acae4ad24ca628213273021453a3a9e17984627e2ab8506d1dcf6e1fabd9ec3123a5f71d2a65295ad6f6f56da2224d4a6e3228237c237447bbf48a6eff1f53e1971503f26c3fb5b9d99d27eca7266529a5f86d75bada7ec10bf314687bfcbf9fa3b09b8e36e73f6d6a154a5ce5ff0&ascene=14&uin=NzE3NzkyOTQx&devicetype=iMac20%2C1+OSX+OSX+14.2.1+build(23C71)&version=13080610&nettype=WIFI&lang=zh_CN&countrycode=CN&fontScale=100&exportkey=n_ChQIAhIQM7tgcovlhXp1y5J%2BRMnhfhL3AQIE97dBBAEAAAAAAFsTBwTm4FMAAAAOpnltbLcz9gKNyK89dVj0MxFZswDc%2Fk646vJPW2S3JFh8H2JhyZXiPbIl%2Bh23CsewrmIoZ4j0D2zMNylC3pLbhu9FIARUKYn%2F0r0OIdHnxesVFpw1qLo6uBJ3zmbsKBVXM05%2B0MiOBIfShfpiIfraK7THzak94U0RdS1flC%2BIDjTb5SmZs9Z4XTyTsN0QXR6NWjAXFeuxnMB4SENMJ8dUR8n08b3DGKtz9rfefn0JRlsX4mGcLvOFsFwg4nk35nl4C3Wgcs4OYociKm5UHabdhWT7%2FWbNNToZLD39eD%2FL4Xo%3D&acctmode=0&pass_ticket=8xG4QKOnJeEFUtXkScVmMqb3omdWJbPuc%2BhN5%2BA7%2FOXj7ex757M2ABNO4GVVnv2T3VtqrC9gZH%2FrU09rrUxJcA%3D%3D&wx_header=0)（от автора Word2Vec）

    Название статьи: Large Language Models: A Survey

    Ссылка на статью: https://arxiv.org/pdf/2402.06196.pdf

2. [Разбор статей GPT, GPT-2, GPT-3 【разбор статьи】](https://www.bilibili.com/video/BV1AF411b7xQ?t=0.0)

3. [Разбор статьи InstructGPT 【разбор статьи·48】](https://www.bilibili.com/video/BV1hd4y187CR?t=0.4)
