# Практикум по большим моделям: редактирование знаний в больших моделях
Введение: методы и инструменты редактирования языковых моделей
> Хотите управлять тем, как языковая модель «помнит» определённые знания? Давайте выберем подходящий метод редактирования, отредактируем конкретное знание и проверим отредактированную модель!

## 1. Цели этого руководства:

- Освоить работу с инструментарием EasyEdit
- Освоить методы редактирования языковых моделей (простейшие)
- Понять, как выбирать разные типы методов редактирования и сценарии их применения

## 2. Подготовка к работе:
### 2.1 Знакомство с EasyEdit

https://github.com/zjunlp/EasyEdit

EasyEdit — это Python-пакет для редактирования языковых моделей, таких как GPT-J, Llama, GPT-NEO, GPT2, T5 и др. Его цель — эффективно изменять поведение языковой модели в отношении конкретного знания, не ухудшая качество на других входных данных, при этом оставаясь простым в использовании и расширении.

EasyEdit интегрирует существующие популярные методы редактирования:
![](./assets/1.png)

### 2.2 Основная архитектура

![](./assets/2.png)
EasyEdit включает единый фреймворк из Editor, Method и Evaluate, которые представляют соответственно сценарий редактирования, технику редактирования и метод оценки.
- Editor: описывает рабочий сценарий, включает редактируемую модель, редактируемое знание, а также другие необходимые гиперпараметры.
- Method: конкретный метод редактирования знаний (например, ROME, MEND и т. д.).
- Evaluate: метрики оценки качества редактирования знаний, включающие надёжность (reliability), обобщаемость (generality), локальность (locality), переносимость (portability).
- Trainer: некоторые методы редактирования требуют определённого процесса обучения, который реализуется модулем Trainer.


## 3. Установка окружения:
```
git clone https://github.com/zjunlp/EasyEdit.git
（опционально）conda create -n EasyEdit python=3.9.7
cd EasyEdit
pip install -r requirements.txt
```


## 4. Пример редактирования
> Цель: изменить «память» модели GPT-2-XL, сменив профессию Месси (Lionel Messi) с футбола на баскетбол (football->basketball).
Шаги:
- Выбрать метод редактирования, подготовить параметры
- Подготовить данные для редактирования знаний
- Создать экземпляр Editor
- Запустить!
Ниже подробно разберём на примере метода ROME:
### 4.1 ROME
Jupyter Notebook: [https://colab.research.google.com/drive/1KkyWqyV3BjXCWfdrrgbR-QS3AAokVZbr?usp=sharing#scrollTo=zWfGkNb9FBJQ]
- Выбираем метод редактирования, готовим параметры
  - В качестве метода редактирования выбираем ROME, готовим параметры, необходимые для ROME и GPT2-XL.
  - Например: alg_name: "ROME", model_name: "./hugging_cache/gpt2-xl" или путь к этой модели локально, "device": номер используемого GPU
  - Остальные параметры можно оставить по умолчанию
![](./assets/3.png)
- Готовим данные для редактирования знаний
    ```
    prompts = ['Question:What sport does Lionel Messi play? Answer:'] # x_e
    ground_truth = ['football'] # y
    target_new = ['basketball'] # y_e
    subject = ['Lionel Messi'] 
    ```
- Создаём экземпляр Editor: передаём подготовленные параметры в класс BaseEditor для инстанцирования и получаем настроенный экземпляр Editor.
    ```
    hparams = ROMEHyperParams.from_hparams('./hparams/ROME/gpt2-xl.yaml')
    editor=BaseEditor.from_hparams(hparams)
    ```
- Запускаем! Вызываем метод edit у editor:
    ```
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        keep_original_weight=False
    )
    ```
![](./assets/4.png)
> Примечание: при первом редактировании конкретной модели будет скачан корпус Wiki, а также вычислены и сохранены состояния (статистики) каждого слоя этой модели (stats_dir: "./data/stats"); они переиспользуются при каждом последующем редактировании. Поэтому первое редактирование занимает некоторое время — при стабильном соединении можно терпеливо подождать.
### 4.2 Проверка и оценка
editor.edit возвращает metrics (вычисляются модулем Evaluate из EasyEdit). В виде:
![](./assets/5.png)
Чтобы получить значения обобщаемости, локальности и переносимости, нужно передать в метод edit данные для оценки.

На примере локальности: это заставит метод edit вычислить метрику локальности, то есть долю правильных ответов модели на locality_inputs.
```
locality_inputs = {
    'neighborhood':{
        'prompt': ['Joseph Fischhof, the', 'Larry Bird is a professional', 'In Forssa, they understand'],
        'ground_truth': ['piano', 'basketball', 'Finnish']
    }
}
metrics, edited_model, _ = editor.edit(
    prompts=prompts,
    ground_truth=ground_truth,
    target_new=target_new,
    locality_inputs=locality_inputs,
    keep_original_weight=False
)
```
Либо можно напрямую сравнить поведение генерации модели до и после редактирования.
```
generation_prompts = [
    "Lionel Messi, the",
    "The law in Ikaalinen declares the language"
]

model = GPT2LMHeadModel.from_pretrained('./hugging_cache/gpt2').to('cuda')
batch = tokenizer(generation_prompts, return_tensors='pt', padding=True, max_length=30)

pre_edit_outputs = model.generate(
    input_ids=batch['input_ids'].to('cuda'),
    attention_mask=batch['attention_mask'].to('cuda'),
    max_new_tokens=3
)
post_edit_outputs = edited_model.generate(
    input_ids=batch['input_ids'].to('cuda'),
    attention_mask=batch['attention_mask'].to('cuda'),
    max_new_tokens=3
)
```


## 5. Масштабное редактирование (опционально)
### 5.1 Batch edit
Несколько записей можно оформить в виде параллельных списков и одновременно передать в метод edit для пакетного редактирования (batch edit); в этом случае MEMIT является наилучшим методом. (https://colab.research.google.com/drive/1P1lVklP8bTyh8uxxSuHnHwB91i-1LW6Z)
```
prompts = ['Question:What sport does Lionel Messi play? Answer:',
            'The law in Ikaalinen declares the language']
ground_truth = ['football', 'Finnish']
target_new = ['basketball', 'Swedish']
subject = ['Lionel Messi', 'Ikaalinen']
```
### 5.2 Тестирование на бенчмарках
- Counterfact
- ZsRE
```
{
    "case_id": 4402,
    "pararel_idx": 11185,
    "requested_rewrite": {
      "prompt": "{} debuted on",
      "relation_id": "P449",
      "target_new": {
        "str": "CBS",
        "id": "Q43380"
      },
      "target_true": {
        "str": "MTV",
        "id": "Q43359"
      },
      "subject": "Singled Out"
    },
    "paraphrase_prompts": [
      "No one on the ground was injured.  v",
      "The sex ratio was 1063. Singled Out is to debut on"
    ],
    "neighborhood_prompts": [
      "Daria premieres on",
      "Teen Wolf was originally aired on",
      "Spider-Man: The New Animated Series was originally aired on",
      "Celebrity Deathmatch premiered on",
      "Æon Flux premiered on",
      "My Super Psycho Sweet 16 premieres on",
      "Daria was released on",
      "Jersey Shore premiered on",
      "Skins was originally aired on",
      "All You've Got premiered on"
    ]
  }
  ```
https://github.com/zjunlp/EasyEdit/blob/main/examples/run_zsre_llama2.py 
