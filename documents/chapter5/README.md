# Практикум по большим моделям: водяные знаки в моделях

Введение: этот раздел знакомит с водяными знаками (watermark) в языковых моделях

> Встраивание в контент, сгенерированный языковой моделью, «водяного знака», незаметного для человека, но обнаруживаемого алгоритмом.

## Цели этого руководства

1. Встраивание водяного знака: внедрение водяного знака при генерации контента языковой моделью
2. Обнаружение водяного знака: определение силы водяного знака в заданном тексте
3. Оценка водяного знака: оценка качества обнаружения у метода водяных знаков
4. Оценка устойчивости (робастности) водяного знака (опционально)



## Подготовка к работе

### 2.1 Знакомство с репозиторием X-SIR

https://github.com/zwhe99/X-SIR

Репозиторий X-SIR содержит реализацию следующего:

- Три алгоритма текстовых водяных знаков: X-SIR, SIR и KGW
- Два метода атак на удаление водяного знака: paraphrase (перефразирование) и translation (перевод)

![img](./assets/x-sir.png)

### 2.2 Подготовка окружения

```Shell
git clone https://github.com/zwhe99/X-SIR && cd X-SIR
conda create -n xsir python==3.10.10
conda activate xsir
pip3 install -r requirements.txt
# [optional] pip3 install flash-attn==2.3.3
```

> Версии в requirements.txt являются рекомендуемыми, а не обязательными.



## Практический пример

> Встраивание водяного знака в контент, генерируемый языковой моделью, с помощью алгоритма KGW

### 3.1 Подготовка данных

Оформляем промпты, которые будут подаваться языковой модели, в виде jsonl-файла:

```JSON
{"prompt": "Ghost of Emmett Till: Based on Real Life Events "}
{"prompt": "Antique Cambridge Glass Pink Decagon Console Bowl Engraved Gold Highlights\n"}
{"prompt": "2009 > Information And Communication Technology Index statistics - Countries "}
......
```

- Каждая строка — это json-объект, содержащий как минимум ключ с именем «prompt»
- В дальнейшем в качестве примера используется файл `data/dataset/mc4/mc4.en.jsonl`. Этот файл содержит всего 500 записей; если вам кажется, что обработка моделью занимает слишком много времени, можно самостоятельно сократить данные.

### 3.2 Встраивание водяного знака

- Выбираем модель и алгоритм водяного знака. Здесь мы выбираем модель `baichuan-inc/Baichuan-7B` и алгоритм водяного знака `KGW`

  - ```Shell
    MODEL_NAME=baichuan-inc/Baichuan-7B
    MODEL_ABBR=baichuan-7b
    WATERMARK_METHOD_FLAG="--watermark_method kgw"
    ```

- Генерируем контент и встраиваем в него водяной знак

  - ```Shell
    python3 gen.py \
        --base_model $MODEL_NAME \
        --fp16 \
        --batch_size 32 \
        --input_file data/dataset/mc4/mc4.en.jsonl \
        --output_file gen/$MODEL_ABBR/kgw/mc4.en.mod.jsonl \
        $WATERMARK_METHOD_FLAG
    ```

  - Эта команда сохраняет сгенерированный моделью контент в выходной файл: `gen/$MODEL_ABBR/kgw/mc4.en.mod.jsonl`

  - Формат выходного файла приведён ниже, где response — это сгенерированный моделью контент:

    - ```JSON
      {"prompt": "Ghost of Emmett Till: Based on Real Life Events ", "response": ".In August if 1955 African American Emmett Louis Till (21)\nThe second part of The Man From Waco, about Dan Millers trial for murdering his friend Michael Capps in a Texas wiener wrastle as I believe the statute says called it then; back at that time that would have surely occurred since Dan kept his pistol in one of those watery doggy bags he keeps around to clean himself with after emptying can into a nearby lake just minutes before committing his crime. If what we read is true thats exactly where Dan left his stolen gun and later used it in the robbery gone wrong which killed two innocent boys when his own accomplice got into an argument over not being paid enough therefore wanting out. This angered Miller whos history of mental instability could be taken one way or another but this criminal act was unavoidable once they entered FBIs hands and some other very powerful law officers who were involved either directly"}
      {"prompt": "Antique Cambridge Glass Pink Decagon Console Bowl Engraved Gold Highlights\n", "response": "An exceptionally fine decorative antique pink decagonal glass side bowl is the perfect example of early art and innovation . Fully engraved, this beautiful English vintage tableware piece exhibits a great degree on craftsmanship! Made in England during the mid 1800's it features three sets of concentric ribbons on the exterior to elegantly highlight an intricate, deep reddish color which evokes warmth and comfort for years to come! This historically significant vase has been featured within numerous museum exhibitions including \"Glass at The Corning Museum\" ; \"The First Half Century\" & a special travelling exhibit called:\" Sight Of Glass: British Cut Glass\" by ibex limited (retailer) as well as \"SIGNALS - Celebrating History In American Silver Through The Articulated Bottle Vessel\" presented at the Corning Museum of Glass 2012 ASA national symposium! We provide our customers with quality phot"}
      {"prompt": "2009 > Information And Communication Technology Index statistics - Countries ", "response": "5/22/2016\nAnnual change of mobile telephone subscriptions in Armenia (per 1 population). 2.2% increase is equivalent to 38 subscriptions per 100 people. Density rank: 121 out of 222.\nCyclist(s)/month(S). Likes bike riding? Take advantage of discount and cheap rental bikes at Rimon Bike Rentals in Yerevan! No advance payments or additional deposits are required. They have a good range of bicycles, including mountainbikes. More on their Facebook page \nYou must know about electric cars. The Renault Fluence KZERO gets it right in the city but I'm not sure what mileage you can expect from it. Still its fun project http://www.renault-kzen.com\nFor more on this and related issues : Armenian Institute for Electronic Governance reports |"}
      ......
      ```



### 3.3 Обнаружение водяного знака

> Обнаружение водяного знака — это, по заданному фрагменту текста, вычисление силы водяного знака в этом фрагменте (z-score).

- Вычисляем силу водяного знака для текста **с водяным знаком**

  - ```python
    python3 detect.py \
        --base_model $MODEL_NAME \
        --detect_file gen/$MODEL_ABBR/kgw/mc4.en.mod.jsonl \
        --output_file gen/$MODEL_ABBR/kgw/mc4.en.mod.z_score.jsonl \
        $WATERMARK_METHOD_FLAG
    ```

- Вычисляем силу водяного знака для текста **без водяного знака**

  - ```python
    python3 detect.py \
        --base_model $MODEL_NAME \
        --detect_file data/dataset/mc4/mc4.en.jsonl \
        --output_file gen/$MODEL_ABBR/kgw/mc4.en.hum.z_score.jsonl \
        $WATERMARK_METHOD_FLAG
    ```

- Формат выходного файла:

  - ```JSON
    {"z_score": 12.105422509165574, "prompt": "Ghost of Emmett Till: Based on Real Life Events ", "response": ".In August if 1955 African American Emmett Louis Till (21)\nThe second part of The Man From Waco, about Dan Millers trial for murdering his friend Michael Capps in a Texas wiener wrastle as I believe the statute says called it then; back at that time that would have surely occurred since Dan kept his pistol in one of those watery doggy bags he keeps around to clean himself with after emptying can into a nearby lake just minutes before committing his crime. If what we read is true thats exactly where Dan left his stolen gun and later used it in the robbery gone wrong which killed two innocent boys when his own accomplice got into an argument over not being paid enough therefore wanting out. This angered Miller whos history of mental instability could be taken one way or another but this criminal act was unavoidable once they entered FBIs hands and some other very powerful law officers who were involved either directly", "biases": null}
    {"z_score": 12.990684249887122, "prompt": "Antique Cambridge Glass Pink Decagon Console Bowl Engraved Gold Highlights\n", "response": "An exceptionally fine decorative antique pink decagonal glass side bowl is the perfect example of early art and innovation . Fully engraved, this beautiful English vintage tableware piece exhibits a great degree on craftsmanship! Made in England during the mid 1800's it features three sets of concentric ribbons on the exterior to elegantly highlight an intricate, deep reddish color which evokes warmth and comfort for years to come! This historically significant vase has been featured within numerous museum exhibitions including \"Glass at The Corning Museum\" ; \"The First Half Century\" & a special travelling exhibit called:\" Sight Of Glass: British Cut Glass\" by ibex limited (retailer) as well as \"SIGNALS - Celebrating History In American Silver Through The Articulated Bottle Vessel\" presented at the Corning Museum of Glass 2012 ASA national symposium! We provide our customers with quality phot", "biases": null}
    {"z_score": 11.455466938203664, "prompt": "2009 > Information And Communication Technology Index statistics - Countries ", "response": "5/22/2016\nAnnual change of mobile telephone subscriptions in Armenia (per 1 population). 2.2% increase is equivalent to 38 subscriptions per 100 people. Density rank: 121 out of 222.\nCyclist(s)/month(S). Likes bike riding? Take advantage of discount and cheap rental bikes at Rimon Bike Rentals in Yerevan! No advance payments or additional deposits are required. They have a good range of bicycles, including mountainbikes. More on their Facebook page \nYou must know about electric cars. The Renault Fluence KZERO gets it right in the city but I'm not sure what mileage you can expect from it. Still its fun project http://www.renault-kzen.com\nFor more on this and related issues : Armenian Institute for Electronic Governance reports |", "biases": null}
    ......
    ```

- Сравните на глаз разницу в силе водяного знака между двумя файлами

### 3.4 Оценка водяного знака
<a name="eval"></a>

- Подаём на вход z-score файлы из обнаружения водяного знака, вычисляем точность обнаружения и строим ROC-кривую

  - ```Shell
    python3 eval_detection.py \
            --hm_zscore gen/$MODEL_ABBR/kgw/mc4.en.hum.z_score.jsonl \
            --wm_zscore gen/$MODEL_ABBR/kgw/mc4.en.mod.z_score.jsonl \
            --roc_curve roc
    
    AUC: 1.000
    
    TPR@FPR=0.1: 0.998
    TPR@FPR=0.01: 0.998
    
    F1@FPR=0.1: 0.999
    F1@FPR=0.01: 0.999
    ```

![img](./assets/curve.png)

## Оценка устойчивости (робастности) водяного знака (опционально)

> После атаки на текст с водяным знаком методами paraphrase и translation повторно оцениваем качество его обнаружения

### 4.1 Подготовка

Мы используем модель gpt-3.5-turbo-1106 для перефразирования (paraphrase) и перевода (translation) текста с водяным знаком. Можно также выбрать другие инструменты по своему усмотрению.

- Устанавливаем apikey OpenAI

  - ```Shell
    export OPENAI_API_KEY=xxxx
    ```

- Меняем в `attack/const.py` значения RPM (requests per min) и TPM (tokens per min)

### 4.2 Проведение атаки (на примере перевода)

- Переводим текст с водяным знаком на китайский

  - ```Shell
    python3 attack/translate.py \
        --input_file gen/$MODEL_ABBR/kgw/mc4.en.mod.jsonl \
        --output_file gen/$MODEL_ABBR/kgw/mc4.en-zh.mod.jsonl \
        --model gpt-3.5-turbo-1106 \
        --src_lang en \
        --tgt_lang zh
    ```

- Повторно оцениваем

  -  см. [3.4](#eval)

- Сравниваем изменение качества водяного знака до и после атаки



## Продвинутые упражнения

- Изучите документацию [X-SIR](https://github.com/zwhe99/X-SIR), научитесь использовать два других алгоритма (X-SIR, SIR) и оцените их качество при разных методах атак
