# Nuclear-IT-hack-24
Репозиторий решения команды AXIOM на отборочном этапе _Nuclear IT Hack 2024_

# Задача

В рамках хакатона команда решала задачу от **МТС Линк**.

Постановка задачи: Разработать систему на основе ИИ, которая анализирует список пользовательских ответов возвращает понятное и интерпретируемое облако слов.

# Команда: AXIOM
- Зворыгин Владимир Андреевич
- Миронов Андрей Михайлович
- Диков Александр Евгеньевич
- Садохин Алексей Александрович
 
# Структура проекта
- **_app_**: streamlit приложение, показывающее демонстрацию работы нашего алгоритма
- **_datasets_**: сгенерированные датасеты для тестирования разработанного алгоритма
- **_experiments_**: ноутбуки с экспериментами и тестированием моделей

# Идея решения

Для того, чтобы получить облако слов из большого числа пользовательских ответов, необходимо эти ответы агрегировать. Агрегировать напрямую слова проблематично. Необходимо провести предобработку слов и преобразовать их в числа, а именно вектора, чтобы далее их объединять/кластеризовать. 

Для работы необходимо иметь датасеты. Для этого были сгенерировали 4 базы ответов разного размера и немного отличающегося наполнения. 

При обработке данные датасета объединились в один массив. В рамках этого массива все буквы приводились к нижнему регистру. Затем было проведено удаление стоп-слов (как на русском, так и на английском), а также лемматизация. Соотношение <оригинальное слово - обработанное слово> было сохранено в словарь для постобработки.

После таких операций в массиве стало много повторяющихся слов и выражений. Мы составили словарь с подсчётом количества каждого объекта. Далее для каждого ключа получившегося словаря были получены эмбеддинги. 

В качестве эмбеддинг моделей были рассмотрены предобученные bert-like модели: 
- [_ru-en-RoSBERTa_](https://huggingface.co/ai-forever/ru-en-RoSBERTa)
- [_ruBert-large_](https://huggingface.co/ai-forever/ruBert-large)
- [_ruElectra-large_](https://huggingface.co/ai-forever/ruElectra-large)

И более простой метод векторизации текста **_word2vec_** - для обучения был взят корпус [русских текстов](https://huggingface.co/Word2vec/wikipedia2vec_ruwiki_20180420_300d). Данный метод показал наибольшую эффективность по скорости, но меньшее качество при построении векторов и дальнейшей кластеризации.

Поэтому для решения в основном были использованы bert-like модели. В рамках модели **_ru-en-RoSBERTa_**, согласно документации, при получении эмбеддингов был использован префикс "_clustering:_ " для повышения точности. 

Для кластеризации были использованы 2 алгоритма - _**AgglomerativeClustering**_ и _**DBSCAN**_, так как эти алгоритмы не требуют задания числа кластеров. Для выбора алгоритма кластеризации на большом датасете были получены эмбеддинги от рассматриваемых моделей, обучены оба алгоритма для создания кластеров, подобраны параметры, чтобы кластеров было не слишком много (состоящих из одного слова) и они были не слишком большие. Для большого датасета это порядка 100 кластеров. Исходя из получаемых кластеров был выбран алгоритм **_AgglomerativeClustering_**, так как слова в получающихся кластерах были более близкими по смыслу. 
С использованием этого алгоритма были проведены эксперименты с другими датасетами и другими моделями. Лучше всех себя показала _**ru-en-RoSBERTa**_. _**ruElectra-large**_ показала результаты примерно на том же уровне, но данная модель работает только с русским языком, в отличие от первой. _**ruBert-large**_ показал худшие результаты среди этих трёх моделей. _**word2vec**_ показал сравнимые результаты. 

Также для визуализации полученные на маленьком датасете эмбеддинги разных моделей мы понизили в размерности до 2 с помощью **_PCA_** и отобразили получающиеся кластеры.  

В результате экспериментов была выбрана модель **_ru-en-RoSBERTa_** (как наиболее точная) и алгоритм **_AgglomerativeClustering_** для кластеризации. Также была рассмотрена обученная нами модель **_word2vec_**. Для них было проведено тестирование производительности на всех датасетах - результаты представлены в экспериментах.  

После кластеризации в каждом кластере выбирается самое популярное слово в рамках всех ответов. Если таких слов несколько, то среди них выбирается наиболее употребимое в языке слово. Далее идёт подсчёт суммы встречаемости в датасете слов одного кластера и приписывается выбранному слову. Получаем словарь: ключ — самое частое слово, значение — сумма частот всех слов в кластере. 
Однако стоит помнить, что слова в данном словаре после лемматизации и удаления стоп-слов. Для получения изначальных форм слов и выражений используем ранее созданный словарь.  

В конечном итоге на основе получившихся данных строится облако слов с использованием библиотеки _**wordcloud**_.

# Демо решения

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/vRde3xHIVh8/0.jpg)](https://www.youtube.com/watch?v=vRde3xHIVh8)

# Разворачивание решения

### **Шаг 1: Клонирование репозитория или подготовка проекта**

Клонируйте репозитории Git:

```commandline
git clone https://github.com/WocherZ/Nuclear-IT-hack-24.git
cd Nuclear-IT-hack-24
```

Перейдите в директорию с приложением:

```commandline
cd app
```

### **Шаг 2: Создание виртуального окружения**

Рекомендуется использовать виртуальное окружение для изоляции зависимостей проекта.

```commandline
python3 -m venv venv
```

Активируйте виртуальное окружение:

- На Windows:
    ```commandline
    venv\Scripts\activate
    ```

- На macOS и Linux:
    ```commandline
    source venv/bin/activate
    ```

### **Шаг 3: Установка зависимостей из requirements.txt**

Убедитесь, что файл requirements.txt находится в корне вашего проекта. Для установки зависимостей выполните:

```commandline
pip install --upgrade pip
pip install -r requirements.txt
```

Это установит все необходимые библиотеки.

### **Шаг 4: Загрузка эмбеддинг модели**

По умолчанию приложение использует модель **_ru-en-RoSBERTa_**. Для смены модели на другую достаточно указать нужное название модели с репозитория Hugging Face.

Для использования **_word2vec_** необходимо выставить соответсвующий флаг в файле _utils/embeddings.py_. А также загрузить [корпус русских текстов](https://huggingface.co/Word2vec/wikipedia2vec_ruwiki_20180420_300d) и положить в директорию _utils_.
Тогда при запуске модель автоматически обучится на этом корпусе.

При первом запуске будет скачиваться/обучаться модель - может потребоваться несколько минут.

### **Шаг 5: Запуск Streamlit приложения**

После установки зависимостей и моделей запустите ваше Streamlit приложение:

```commandline
streamlit run app.py
```

После выполнения команды, Streamlit запустит локальный сервер, и в командной строке отобразится URL (обычно http://localhost:8501), по которому можно открыть приложение в браузере.
