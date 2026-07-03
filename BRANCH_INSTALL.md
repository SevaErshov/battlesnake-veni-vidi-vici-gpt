# Заливка в отдельную ветку

Рекомендуемая ветка: `feature/hardml-svm-safe`.

```powershell
git checkout main
git pull origin main
git checkout -b feature/hardml-svm-safe
```

Скопируй содержимое этой папки в корень форка с заменой файлов, затем:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python smoke_test.py
python backend.py
```

Если smoke-test прошел, пушим:

```powershell
git add .
git commit -m "Add HardML SVM Safe Battlesnake"
git push origin feature/hardml-svm-safe
```
