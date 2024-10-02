import json
import requests

import os
from dotenv import load_dotenv

load_dotenv()



def analyze_word_frequency_gigachat(word_frequencies, question):
    """
    Анализирует словарь с частотой слов с помощью GigaChat, генерируя текстовый ответ на заданный вопрос.

    Args:
        word_frequencies (dict): Словарь, где ключи - слова, а значения - их частоты.
        question (str): Вопрос для GigaChat.

    Returns:
        str: Текстовый ответ GigaChat.
    """

    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

    payload={
        'scope': 'GIGACHAT_API_PERS'
    }
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': f'{os.environ.get("GIGACHAT_ID")}',
        'Authorization': f'Basic {os.environ.get("GIGACHAT_KEY")}'
    }

    response = requests.request("POST", url, headers=headers, data=payload, verify=False)

    access_token = json.loads(response.text)["access_token"]

    # Преобразование словаря частотности слов в строку для промпта
    word_frequencies_str = "\n".join(f"{word}: {freq}" for word, freq in word_frequencies.items())

    # Создание промпта для GigaChat
    prompt = (
        f"Вот распределение слов по частоте:\n\n{word_frequencies_str}\n\n"
        f"Вопрос: {question}\n\n"
        f"Ответ:"
    )

    url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

    payload = json.dumps({
        "model": "GigaChat",
        "messages": [
            {
            "role": "system",
            "content": "Ты профессиональный аналитик. Тебе необходимо проанализировать распределение по частоте ответов на вопрос. Выдай краткую аналитику по предоставленным результатам. Напиши 2-3 предложения, без перечислений."
            },
            {
            "role": "user",
            "content": f"{prompt}"
            }
        ],
        "stream": False,
        "update_interval": 0
    })
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }

    response = requests.request("POST", url, headers=headers, data=payload, verify=False)

    # Обработка ответа от GigaChat
    if response.status_code == 200:
        response_data = response.json()
        return response_data['choices'][0]['message']['content'].strip()
    else:
        print(f"Ошибка при обращении к GigaChat API: {response.text}")
        return None