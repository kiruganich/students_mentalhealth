# Student Mental Health Analysis

## Overview

This project studies how students’ lifestyle habits relate to their academic performance (**CGPA**) and mental health (**depression risk**).  
На основе одного табличного датасета строятся две модели: регрессия для CGPA и классификация риска депрессии, с полным ML‑пайплайном (EDA → предобработка → feature engineering → обучение → оценка).

## Dataset

- **Source:** `student_lifestyle_100k.csv`  
- **Origin:** Kaggle — Student Depression and Lifestyle (100k data)  
- **Size:** 100 000 записей, 11 исходных признаков  
- **Raw features:**
  - `StudentID` — идентификатор студента
  - `Age` — возраст
  - `Gender` — пол
  - `Department` — факультет (Science, Engineering, Medical, Arts, Business)
  - `CGPA` — средний балл успеваемости
  - `SleepDuration` — часы сна в сутки
  - `StudyHours` — часы учёбы в день
  - `SocialMediaHours` — часы в соцсетях
  - `PhysicalActivity` — уровень/частота физической активности
  - `StressLevel` — уровень стресса (1–10)
  - `Depression` — булевый флаг риска депрессии

- **Target variables:**
  - `CGPA` — непрерывная величина (регрессионная задача)
  - `Depression` — бинарная метка (задача классификации)

Дополнительно в процессе работы конструируются новые признаки, отражающие баланс сна/учёбы и общую нагрузку.

## Machine Learning Tasks

### 1. CGPA Predictor (Regression)

Задача: спрогнозировать академическую успеваемость (CGPA) студента на основе его режима сна, учёбы и других характеристик образа жизни.

- **Algorithm:** `LinearRegression` (через `sklearn.pipeline.Pipeline`)
- **Features (примерный набор):**
  - `SleepDuration`, `StudyHours`, `SocialMediaHours`, `PhysicalActivity`, `StressLevel`
  - категориальные признаки `Gender`, `Department` (One‑Hot Encoding)
  - инженерные признаки: `StudySleepRatio`, `TotalLoad`, `WellnessScore`
- **Preprocessing:**
  - числовые признаки — масштабирование `StandardScaler`
  - категориальные признаки — OneHotEncoding / dummy‑переменные
  - предобработка инкапсулирована в `ColumnTransformer` + `Pipeline`

Основные метрики регрессии (на валидации / тесте):

- \(R^2\) — доля объяснённой дисперсии  
- `MAE`, `RMSE` — ошибки прогноза в исходном масштабе CGPA  

(Конкретные значения см. в итоговых ячейках ноутбука с оценкой модели.)

### 2. Depression Classifier (Classification)

Задача: по образу жизни и признакам стресса предсказать, относится ли студент к группе риска по депрессии (`Depression = True/False`).

- **Algorithm:** `SGDClassifier` с `loss='log_loss'` (эквивалент логистической регрессии с SGD)
- **Features (примерный набор):**
  - `Age`, `CGPA`, `SleepDuration`, `StudyHours`, `SocialMediaHours`
  - `PhysicalActivity`, `StressLevel`
  - категориальные фичи `Gender`, `Department` (One‑Hot Encoding)
  - инженерные признаки: `StudySleepRatio`, `TotalLoad`, `WellnessScore`
- **Preprocessing:**
  - числовые признаки — `StandardScaler`
  - категориальные — OneHotEncoding
  - всё оформлено через `Pipeline` и `ColumnTransformer`
- **Training setup:**
  - `class_weight='balanced'` для компенсации возможного дисбаланса классов
  - `early_stopping=True`, `validation_fraction=0.1` для контроля переобучения
  - сеточный поиск по:
    - `learning_rate='constant'`, `eta0 ∈ {0.001, 0.01, 0.05, 0.1}`
    - `max_iter ∈ {50, 100, 200, 500}`

**Key metrics (на валидации / тесте):**

- `Accuracy`
- `Precision`, `Recall`, `F1-score`
- `ROC-AUC`
- при необходимости — `PR-AUC`, ROC/PR‑кривые

Грид‑поиск позволяет подобрать сочетание шага обучения и числа эпох, при котором достигается лучший F1 и ROC‑AUC без переобучения.

## Pipeline

Полный ML‑пайплайн реализован внутри `notebook.ipynb` и включает следующие этапы:

1. **Exploratory Data Analysis (EDA)**
   - описание структуры датасета, типов данных и базовой статистики
   - распределения числовых признаков (histogram, boxplot)
   - анализ категориальных признаков (частоты полов и факультетов)
   - проверка логических аномалий (диапазоны возраста, CGPA, часов сна и активности)
   - корреляционная матрица между признаками, связь CGPA и Depression с lifestyle‑факторами

2. **Feature Engineering**
   - создание новых признаков:
     - `StudySleepRatio` — отношение часов учёбы к часам сна
     - `TotalLoad` — суммарная нагрузка (учёба + соцсети + активность)
     - `WellnessScore` — агрегированный индекс благополучия (сон, активность, стресс)
   - анализ полезности новых фичей (корреляции, распределения, влияние на качество моделей)

3. **Preprocessing**
   - переименование колонок для удобства (`Student_ID` → `StudentID`, `Sleep_Duration` → `SleepDuration` и т.д.)
   - проверка и при необходимости корректировка диапазонов значений
   - кодирование категориальных признаков (Gender, Department) в one‑hot представление
   - масштабирование числовых признаков `StandardScaler`
   - оформление всей предобработки в `sklearn`‑пайплайнах для предотвращения утечки данных

4. **Train/Validation/Test Split**
   - разбиение данных на:
     - train: 60 %
     - validation: 20 %
     - test: 20 %
   - для классификации — при необходимости стратификация по `Depression`
   - фиксированный `RANDOM_STATE = 42` для воспроизводимости

5. **Model Training**
   - обучение модели линейной регрессии для задачи прогнозирования CGPA
   - обучение логистической модели (`SGDClassifier`) для задачи классификации риска депрессии
   - перебор гиперпараметров `learning rate` и `max_iter` (число эпох) для классификатора
   - использование `early_stopping` для остановки по качеству на внутренней валидации

6. **Evaluation & Diagnostics**
   - для регрессии: `R²`, `MAE`, `RMSE`, анализ остатков
   - для классификации: `Accuracy`, `Precision`, `Recall`, `F1`, `ROC-AUC`
   - построение ROC/PR‑кривых
   - сравнение метрик на train vs validation/test для демонстрации отсутствия переобучения
   - интерпретация значимых признаков и их влияния на прогноз

7. **Self‑Scoring**
   - добавление собственной строки со значениями признаков студента
   - прогон этой строки через обученные модели, оценка собственного CGPA/риска депрессии (бонус из ТЗ)

## Technologies & Libraries

- **Python 3**
- **pandas** — работа с табличными данными
- **NumPy** — численные вычисления
- **matplotlib**, **seaborn** — визуализация и EDA
- **scikit-learn** — `Pipeline`, `ColumnTransformer`, `StandardScaler`, `LinearRegression`, `SGDClassifier`, метрики (`r2_score`, `mean_absolute_error`, `roc_auc_score`, `classification_report` и др.)

## Project Structure

```bash
├── notebook.ipynb               # Основной ноутбук с полным анализом и моделями
├── student_lifestyle_100k.csv   # Исходный датасет
└── README.md                    # Описание проекта (этот файл)
```

## Reproducibility

Все шаги анализа и обучения содержатся в `notebook.ipynb` и могут быть воспроизведены последовательным выполнением ячеек сверху вниз.

- Используется фиксированный `RANDOM_STATE = 42` для сплитов и моделей.
- Предобработка, feature engineering и обучение оформлены в `sklearn`‑пайплайны, что обеспечивает повторяемость результатов и защищает от утечки данных.
- Для корректной работы достаточно установить стандартный стек Python (pandas, NumPy, matplotlib, seaborn, scikit‑learn) и запустить ноутбук в любой совместимой среде (Jupyter / VS Code / Colab).
