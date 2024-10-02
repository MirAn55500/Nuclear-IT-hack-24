# Experiments

Результаты тестирования производительности. Показано время работы в секундах:

| Датасет | Кол-во ответов | Word2vec | BERT embeddings on CPU | BERT embeddings on GPU |
|---------|----------------|----------|------------------------|------------------------|
| Small   | 100            | 1.7      | 15.7                   | 5.7                    |
| Medium  | 500            | 1.8      | 70.6                   | 14.7                   |
| Big     | 2500           | 3.9      | 371                    | 60.6                   |
| Large   | 5000           | 6.9      | 723                    | 119.6                  |

Результаты кластеризации и проецирования на 2 оси:  

## ruElectra-large
<p align="center">
  <img src="https://github.com/user-attachments/assets/16a03791-3453-489e-8c47-134639ca86be" alt="ruElectra-large" width="500"/>
  <br>
</p>

Кластеры слов:
![image](https://github.com/user-attachments/assets/e78ef5fc-c03b-4d99-ace4-5157e298fc92)

## ruBert-large
<p align="center">
  <img src="https://github.com/user-attachments/assets/9f12e1e3-1df8-4809-9f0c-ab9e34427f19" alt="ruBert-large" width="500"/>
  <br>
</p>

Кластеры слов:
![image](https://github.com/user-attachments/assets/e6553802-f263-47fc-9853-651b03816f8d)

## ru-en-RoSBERTa
<p align="center">
  <img src="https://github.com/user-attachments/assets/b8dcda21-80d2-4dfb-8448-eb8fbdbd2b04" alt="ru-en-RoSBERTa" width="500"/>
  <br>
</p>

Кластеры слов:
![image](https://github.com/user-attachments/assets/5e2dfd18-5bdf-47cd-a119-2f8c6c15fdfa)

# Заключение
Лучшей моделью в ходе экспериментов стала ru-en-RoSBERTa
