[README.md](https://github.com/user-attachments/files/25497510/README.md)
# 💊 Классификация фармацевтических препаратов с CNN

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

**Классификация лекарственных препаратов по фотографиям таблеток с помощью свёрточных нейронных сетей**

[📓 Открыть ноутбук](./solution.ipynb) · [📊 Результаты](#-результаты) · [🚀 Быстрый старт](#-быстрый-старт)

</div>

---

## 📌 О проекте

Данный проект решает задачу **многоклассовой классификации изображений** лекарственных препаратов по фотографиям. Модель обучена распознавать более 80 наименований препаратов.

Практическое применение подобной системы:
- 🏥 Автоматизация аптечного учёта
- 📦 Контроль складских запасов
- 🤖 Помощь пациентам в идентификации препаратов

---

## 🧠 Архитектура

Проект основан на технике **Transfer Learning** с использованием предобученной свёрточной нейронной сети (CNN).

```
Input Image (224×224×3)
        │
        ▼
┌──────────────────┐
│  Backbone (CNN)  │  ← предобученная сеть (MobileNet)
│  Feature Extract │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Classifier Head │
│  FC              │
└────────┬─────────┘
         │
         ▼
    Class Prediction
    (80+ классов)
```

---

## 📊 Результаты

| Метрика | Значение |
|---------|----------|
| **Accuracy** | **76%** |
| Всего классов | 80+ |
| Тестовых примеров | 504 |
| Примеров на класс | 6 |

### 🏆 Лучшие классы (F1 = 1.00)

| Препарат | Precision | Recall | F1 |
|----------|-----------|--------|----|
| acc_long_600_mg | 1.00 | 1.00 | **1.00** |
| advil_ultra_forte | 1.00 | 1.00 | **1.00** |
| betaloc_50_mg | 1.00 | 1.00 | **1.00** |
| bila_git | 1.00 | 1.00 | **1.00** |
| c_vitamin_teva_500_mg | 1.00 | 1.00 | **1.00** |

### ⚠️ Сложные классы (низкий F1)

| Препарат | Precision | Recall | F1 | Причина |
|----------|-----------|--------|----|---------|
| atorvastatin_teva_20_mg | 0.50 | 0.17 | **0.25** | Визуально похож на `atoris_20_mg` |
| aspirin_ultra_500_mg | 0.50 | 0.83 | **0.62** | Типичный дизайн для АСК-препаратов |
| algopyrin_500_mg | 0.56 | 0.83 | **0.67** | Схожее оформление с другими анальгетиками |

---

## 🗂️ Структура проекта

```
CNN-Practice/
│
├── 📁 best_model/
│   └── best_model.pt              # Веса лучшей обученной модели
│
├── 📁 dataset/                    # Датасет (не включён в репозиторий)
│   └── ...
│
├── 📁 downloaded_model/
│   └── tf_mobilenetv3_small_075.pth  # Предобученный MobileNetV3
│
├── 📁 src/
│   ├── data_handle.py             # Загрузка и аугментация данных
│   ├── model.py                   # Архитектура и конфигурация модели
│   └── one_epoch.py               # Логика одной эпохи обучения/валидации
│
├── ⚙️  config.yaml                # Гиперпараметры и конфигурация
├── 📋 requirements.txt            # Зависимости проекта
├── 📓 solution.ipynb              # Основной ноутбук с решением
├── 🔒 .gitignore
└── 📄 README.md
```

---

## 🚀 Быстрый старт

### 1. Клонирование репозитория

```bash
git clone https://github.com/Irtair/CNN-Practice.git
cd CNN-Practice
```

### 2. Установка зависимостей

```bash
pip install torch torchvision
pip install numpy matplotlib scikit-learn
pip install jupyter pillow tqdm
```

### 3. Запуск ноутбука

```bash
jupyter notebook solution.ipynb
```

---

## 🔧 Стек технологий

| Категория | Инструмент |
|-----------|------------|
| Язык | Python 3.9+ |
| Deep Learning | PyTorch / Torchvision |
| Метрики | scikit-learn |
| Среда | Jupyter Notebook |
| Аугментации | torchvision.transforms.v2 |

---

## 📈 Идеи для улучшения

- [ ] **Больше данных** — увеличить выборку (сейчас только 6 примеров на класс)
- [ ] **Аугментации** — добавить контрастность
- [ ] **Более мощный backbone** — попробовать EfficientNet-B4, ConvNeXt, ViT
- [ ] **Confusion Matrix** — анализ конкретных пар путаемых классов
