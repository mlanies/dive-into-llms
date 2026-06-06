# Практикум по большим моделям: построение GUI-агента

## Цели этого руководства
1. Познакомиться с технологическими подходами и текущим состоянием исследований в области GUI-агентов

2. Попробовать метод построения адаптивного GUI-агента для человеко-машинного взаимодействия на основе открытой модели Qwen2-VL-7B с опорой на датасет OS-Kairos

3. Попробовать выполнить инференс на построенном GUI-агенте

## 1. Подготовка к работе
### 1.1 Знакомство с технологическими подходами и состоянием исследований в области GUI-агентов
Изучите руководство: [[Слайды](https://github.com/Lordog/dive-into-llms/blob/main/documents/chapter8/GUIagent.pdf)]

### 1.2 Что такое адаптивный GUI-агент для человеко-машинного взаимодействия
Справочная статья: [[Paper](https://arxiv.org/abs/2503.16465)]

### 1.3 Подготовка датасета
В этом руководстве на примере OS-Kairos мы строим и запускаем простой GUI-агент на основе открытого в этой работе датасета.

Скачайте датасет по ссылке из файла README.md в [[Data](https://github.com/Wuzheng02/OS-Kairos)] и распакуйте его в окружение.

Пример формата данных приведён ниже:
```json
    {
        "task": "Открыть NetEase Cloud Music, найти «Shape of You» и воспроизвести эту песню.",
        "image_path": "/data1/wuzh/cloud_music/images/1736614680.6518524_1.png",
        "list": [
            " Открыть NetEase Cloud Music  ",
            " Нажать на поисковую строку вверху главной страницы  ",
            " Ввести: Shape of You  ",
            " Выбрать правильный результат поиска  ",
            " Нажать на название песни для воспроизведения  "
        ],
        "now_step": 1,
        "previous_actions": [
            "CLICK <point>[[381,367]]</point>"
        ],
        "score": 5,
        "osatlas_action": "CLICK <point>[[454,87]]</point>",
        "teacher_action": "CLICK <point>[[500,100]]</point>",
        "success": false
    },
```
### 1.4 Подготовка модели
Скачайте веса модели Qwen2-VL-7B из [[Model](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)].

## 1.5 Подготовка кода для обучения с учителем (SFT)
В этом руководстве для обучения Qwen2-VL-7B с учителем (supervised fine-tuning) используется LLaMa-Factory, поэтому сначала скачайте исходный код из [[Code](https://github.com/hiyouga/LLaMA-Factory/)].

## 2. Построение GUI-агента
### 2.1 Предобработка данных
Сохраните приведённый ниже код как get_sharpgpt.py, укажите в нём пути к Kairos_train.json и к предполагаемому месту сохранения обработанного json обучающего набора, затем запустите этот файл.


```python
import json


with open('', 'r', encoding='utf-8') as f:
    data = json.load(f)


preprocessed_data = []
for item in data:
    task = item.get("task", "")
    previous_actions = item.get("previous_actions", "") 
    image_path = item.get("image_path","")
    prompt_text = f"""
    You are now operating in Executable Language Grounding mode. Your goal is to help users accomplish tasks by suggesting executable actions that best fit their needs and give a score. Your skill set includes both basic and custom actions:

    1. Basic Actions
    Basic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability. 
    Basic Action 1: CLICK 
        - purpose: Click at the specified position.
        - format: CLICK <point>[[x-axis, y-axis]]</point>
        - example usage: CLICK <point>[[101, 872]]</point>

    Basic Action 2: TYPE
        - purpose: Enter specified text at the designated location.
        - format: TYPE [input text]
        - example usage: TYPE [Shanghai shopping mall]

    Basic Action 3: SCROLL
        - Purpose: SCROLL in the specified direction.
        - Format: SCROLL [direction (UP/DOWN/LEFT/RIGHT)]
        - Example Usage: SCROLL [UP]

    2. Custom Actions
    Custom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.

    Custom Action 1: PRESS_BACK
        - purpose: Press a back button to navigate to the previous screen.
        - format: PRESS_BACK
        - example usage: PRESS_BACK

    Custom Action 2: PRESS_HOME
        - purpose: Press a home button to navigate to the home page.
        - format: PRESS_HOME
        - example usage: PRESS_HOME

    Custom Action 3: ENTER
        - purpose: Press the enter button.
        - format: ENTER
        - example usage: ENTER

    Custom Action 4: IMPOSSIBLE
        - purpose: Indicate the task is impossible.
        - format: IMPOSSIBLE
        - example usage: IMPOSSIBLE

    In most cases, task instructions are high-level and abstract. Carefully read the instruction and action history, then perform reasoning to determine the most appropriate next action.

    And I hope you evaluate your action to be scored, giving it a score from 1 to 5. 
    A higher score indicates that you believe this action is more likely to accomplish the current goal for the given screenshot.
    1 means you believe this action definitely cannot achieve the goal.
    2 means you believe this action is very unlikely to achieve the goal.
    3 means you believe this action has a certain chance of achieving the goal.
    4 means you believe this action is very likely to achieve the goal.
    5 means you believe this action will definitely achieve the goal.

    And your final goal, previous actions, and associated screenshot are as follows:
    Final goal: {task}
    previous actions: {previous_actions}
    screenshot:<image>
    Your output must strictly follow the format below, and especially avoid using unnecessary quotation marks or other punctuation marks.(where osatlas action must be one of the action formats I provided and score must be 1 to 5):
    action:
    score:
    """

    action = item.get("osatlas_action", "")
    score = item.get("score", "")
    preprocessed_item = {
        "messages": [
            {
                "role": "user",
                "content": prompt_text,
            },
            {
                "role": "assistant",
                "content": f"action: {action}\nscore:{score}",
            }
        ],
        "images":[image_path]
    }
    preprocessed_data.append(preprocessed_item)


with open('', 'w', encoding='utf-8') as f:
    json.dump(preprocessed_data, f, ensure_ascii=False, indent=4)
```

После этого шага мы успешно преобразуем обучающий набор OS-Kairos в данные формата sharpgpt, удобные для следующего этапа обучения; предобработка данных завершена.

## 2.2 Обучение с учителем (SFT)
На предыдущем шаге мы уже получили датасет Kairos в формате, подходящем для обучения в LLaMa-Factory. Теперь нужно модифицировать LLaMa-Factory, чтобы зарегистрировать датасет и настроить параметры обучения.
Сначала измените data/dataset_info.json, добавив следующее содержимое для регистрации датасета:
```json
"Karios" :{
  "file_name": "Karios_qwenscore.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "messages",
    "images": "images"
  },
  "tags": {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant"
  }
},   
```
Далее измените /examples/train_full/qwen2vl_full_sft.yaml для настройки параметров обучения:
```yaml
### model
model_name_or_path: 
stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json

### dataset
dataset: Karios
template: qwen2_vl
cutoff_len: 4096
max_samples: 9999999
#max_samples: 999
overwrite_cache: true
preprocessing_num_workers: 4

### output
output_dir: 
logging_steps: 10
save_steps: 60000
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 2
learning_rate: 1.0e-5
num_train_epochs: 5.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### eval
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 20000
```
Выше приведён пример конфигурации: model_name_or_path — это путь к Qwen2-VL-7B, output_dir — путь для сохранения чекпойнтов и логов.
После настройки можно переходить к инференсу; пример командной строки:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen2vl_full_sft.yaml
```
Здесь требуется минимум 3 видеокарты A100 на 80 ГБ.

## 3. Проверка инференса OS-Kairos
После завершения обучения можно выполнять инференс; для инференса есть много способов на выбор — можно самостоятельно написать код инференса с помощью библиотек transformers и torch, либо использовать LLaMa-Factory для инференса «в одну команду». Ниже показано, как выполнить инференс «в одну команду» с помощью LLaMa-Factory.
Сначала измените /examples/inference/qwen2_vl.yaml; пример приведён ниже
```yaml
model_name_or_path: 
template: qwen2_vl
```
Здесь model_name_or_path — это путь к чекпойнту после обучения.
Затем можно выполнить инференс «в одну команду» с помощью
```
CUDA_VISIBLE_DEVICES=0 FORCE_TORCHRUN=1 llamafactory-cli webchat examples/inference/qwen2_vl.yaml
```
Вы можете взять изображение из тестового набора OS-Kairos или скриншот с личного телефона, объединить его с текстовым промптом в формате из get_sharpgpt.py и подать агенту; после инференса OS-Karios выдаст action, который он считает нужным выполнить сейчас, и оценку уверенности (confidence score) для этого действия. Чем ниже оценка уверенности, тем сильнее текущая инструкция выходит за границы возможностей OS-Kairos и тем больше требуется вмешательство человека для человеко-машинного взаимодействия.
