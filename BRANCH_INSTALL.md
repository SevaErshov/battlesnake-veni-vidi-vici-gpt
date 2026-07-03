# Как залить этот вариант в отдельную ветку

Из корня вашего fork-репозитория:

```powershell
git checkout main
git pull origin main
git checkout -b feature/expert-safety

# затем скопируй файлы из этой папки в корень репозитория с заменой backend.py/logic.py/render.yaml и т.д.

git add .
git commit -m "Add HardML Expert Battlesnake"
git push origin feature/expert-safety
```

Локальная проверка:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python smoke_test.py
python backend.py
```

Модель по умолчанию: `expert`.
