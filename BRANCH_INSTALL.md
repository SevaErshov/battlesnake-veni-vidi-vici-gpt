# Как залить этот вариант в отдельную ветку

Из корня вашего fork-репозитория:

```powershell
git checkout main
git pull origin main
git checkout -b feature/hardml-ensemble

# затем скопируй файлы из этой папки в корень репозитория с заменой backend.py/logic.py/render.yaml и т.д.

git add .
git commit -m "Add HardML Ensemble Battlesnake"
git push origin feature/hardml-ensemble
```

Локальная проверка:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python smoke_test.py
python backend.py
```

Модель по умолчанию: `ensemble`.
