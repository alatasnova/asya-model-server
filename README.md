# Подготовка
## Как запустить сервер?
У вас должен быть установлен Python, желательно версии 3.12\
После нужно установить минимальные зависимости:
```bash
pip install -c requirements_min.txt
```
Потом нужно запустить файл start_server.py
```bash
python start_server.py 5000
```
Сервер запустится на localhost:5000

## Как обучать модель?
У вас должен быть установлен Python, желательно версии 3.12\
После нужно установить все зависимости:
```bash
pip install -c requirements_full.txt
```
Измените dataset.json и MODEL_CONFIG.ini по своему усмотрению, а после обучите модель.
```bash
python train.py
```
Не забудьте ввести Y после окончания обучения, что бы сохранить модель.

# Использование
## Server API
### Модель распознавания
Пример запроса:\
GET http://localhost:port/recognition?text=включи%20музыку  
Ответ:
```json
{"error":null,"result":{"answer":"turn_on_music","confidence":0.840986967086792}}
```
## Train
крч в конфиге MODEL_CONFIG.ini надо потыкать чета
