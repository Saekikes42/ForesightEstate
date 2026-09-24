# Git-схема проекта

## Ветки

```text
main       — стабильная версия для сдачи
└── develop — общая интеграционная ветка
    ├── feature/eda             — EDA и визуализация
    ├── feature/preprocessing   — очистка и новые признаки
    ├── feature/models          — модели, tuning и метрики
    ├── feature/app             — Streamlit: форма, дашборд, справка
    └── feature/docs            — README и отчёт
```

## Распределение работы

- Участник 1: `feature/eda`, `feature/preprocessing`, `feature/models`.
- Участник 2: `feature/app`, `feature/docs`.
- Интеграция приложения начинается после сохранения модели и препроцессора.

## Первый запуск

Репозиторий в папке проекта пока не инициализирован. Один участник выполняет:

```bash
cd C:/ForesightEstate
git init
git add .
git commit -m "chore: initial project setup"
git branch -M main
git remote add origin <URL-репозитория>
git push -u origin main
git switch -c develop
git push -u origin develop
```

Второй участник:

```bash
git clone <URL-репозитория>
cd ForesightEstate
git switch develop
```

## Рабочий цикл

```bash
git switch develop
git pull origin develop
git switch -c feature/название-задачи
git add .
git commit -m "feat: описание изменения"
git push -u origin feature/название-задачи
```

После этого создаётся Pull Request:

```text
feature/* ── Pull Request ──> develop ── проверка ──> main ──> v1.0.0
```

Правила:

1. Не коммитить напрямую в `main` и `develop`.
2. Каждый Pull Request проверяет второй участник.
3. Один коммит — одно логическое изменение.
4. Перед слиянием запускать приложение и проверять тесты.
5. Исходные данные не перезаписывать, секреты и `.venv` не добавлять.

## Финальный релиз

```bash
git switch main
git pull origin main
git merge --no-ff develop -m "release: версия 1.0.0"
git tag -a v1.0.0 -m "Первая готовая версия"
git push origin main --tags
```
