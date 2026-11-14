# ЛР1 — Методы оптимизации (Линейное программирование), вариант 6

Проект решает задачу планирования выработки электроэнергии на 24 часа (5 станций) как непрерывную LP-модель:
минимизация стоимости при выполнении почасового спроса, лимитов мощности и суточных энергетических лимитов.

## Структура

- `run_lab1_variant6.py` — единая точка запуска (генерация данных, решение, проверка, сценарии, отчётные артефакты).
- `src/data_generation.py` — генерация и загрузка данных варианта.
- `src/lp_model.py` — построение LP и вызов HiGHS через `scipy.optimize.linprog`.
- `src/verification.py` — независимая проверка на NumPy (допустимость + пересчёт целевой функции).
- `src/reporting.py` — таблицы решения, теневые цены, график.
- `data/variant6_base_data.json` — файл данных базового сценария.
- `outputs/` — результаты решения, логи, CSV, график.
- `report/lab1_variant6_report.md` — краткий отчёт по лабораторной.

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Запуск (одной командой)

```bash
python run_lab1_variant6.py --n 17 --q 1 --method highs-ds
```

После запуска все артефакты сохраняются в `outputs/`.

Примечания:
- если `--seed` не передан, seed вычисляется по ТЗ: `10000 + 100*Q + N`;
- `--method highs-ds` — dual simplex, `--method highs-ipm` — interior point;
- presolve включен по умолчанию (отключение: `--no-presolve`).

## Пример ключевых файлов результата

- `outputs/solver_log_base.json` — статус солвера для базового сценария.
- `outputs/solution_base.csv` — почасовая выработка по станциям.
- `outputs/verification_results.csv` — независимая проверка по сценариям.
- `outputs/scenario_results.csv` — базовый/дефицитный/стрессовый сценарии.
- `outputs/shadow_prices_base.csv` — теневые цены ограничений.
- `outputs/dispatch_base.png` — график оптимального почасового диспетчерского плана.

## Дополнительное задание (звёздочка)

Сравнение dual simplex и interior point:

```bash
python run_star_simplex_vs_ipm.py --n 17 --q 1
```

Выходные файлы:
- `outputs/star_simplex_vs_ipm.csv`
- `outputs/star_simplex_vs_ipm.md`
