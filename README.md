# AgriAI – AI Agriculture Project (Flask + MongoDB + TensorFlow + Scikit-learn)

## 1. Project Structure

```
AI-Crop/
├── app.py                  # App factory, blueprint registration, error handlers, seed data
├── config.py                # Centralized configuration (paths, upload limits, model classes)
├── db.py                     # MongoDB connection + collections + indexes
├── requirements.txt
├── models/
│   ├── disease_model.h5      # Your trained TensorFlow/Keras model (place here)
│   └── crop_model.pkl        # Your trained Scikit-learn model (place here)
├── routes/
│   ├── __init__.py
│   ├── auth_routes.py         # /register, /login, /logout
│   ├── crop_routes.py         # /predict-crop
│   ├── disease_routes.py      # /predict-disease
│   └── dashboard_routes.py    # /dashboard, /history
├── utils/
│   ├── __init__.py
│   ├── auth_utils.py          # bcrypt hashing + login_required decorator
│   ├── validators.py          # input validation helpers
│   └── model_loader.py        # lazy-loads ML models + prediction logic
├── templates/
│   ├── base.html, login.html, register.html, dashboard.html,
│   ├── crop.html, disease.html, history.html, 404.html, 500.html
├── static/css/style.css
└── uploads/                    # uploaded leaf images stored here
```

## 2. File-by-File Explanation

- **app.py** – Application factory (`create_app`). Registers all blueprints, creates the upload folder, initializes MongoDB indexes, seeds `disease_solutions` with sample remedy data if empty, defines `/`, `/uploads/<filename>`, and error handlers (404/413/500).
- **config.py** – All configurable constants: secret key, upload folder, allowed extensions, max upload size, model file paths, and the ordered list of disease class names (must match your model's training label order).
- **db.py** – Creates the `MongoClient`, selects `AgriAI_DB`, exposes the 4 collections, and `init_indexes()` for uniqueness/performance.
- **utils/auth_utils.py** – `hash_password`/`verify_password` (bcrypt) and `login_required` decorator that redirects unauthenticated users to `/login`.
- **utils/validators.py** – Email/username/password format checks, file extension check, and crop-input validation/parsing (with range checks for pH and humidity).
- **utils/model_loader.py** – Loads `disease_model.h5` (Keras) and `crop_model.pkl` (Scikit-learn) once (singleton), and provides `predict_disease(image_path)` and `predict_crop(features)`.
- **routes/auth_routes.py** – Register (validates + hashes password + uniqueness check), Login (verifies hash, sets `session['user_id']`/`session['username']`), Logout (clears session).
- **routes/disease_routes.py** – Validates uploaded image, saves with a UUID filename, runs prediction, looks up remedy in `disease_solutions`, stores result in `disease_predictions`, renders result.
- **routes/crop_routes.py** – Validates 7 numeric inputs, runs prediction, stores result in `crop_predictions`, renders result.
- **routes/dashboard_routes.py** – `/dashboard` shows user info, last 5 crop & disease predictions, and total counts. `/history` shows full history tables.
- **templates/** – Bootstrap 5 based UI, extending `base.html` (navbar + flash messages).

## 3. MongoDB Schema

### `users`
```json
{
  "_id": ObjectId,
  "username": "string (unique)",
  "email": "string (unique, lowercase)",
  "password": "bcrypt hash string",
  "created_at": "datetime (UTC)"
}
```

### `crop_predictions`
```json
{
  "_id": ObjectId,
  "user_id": "string (User._id as string)",
  "inputs": {
    "N": float, "P": float, "K": float,
    "temperature": float, "humidity": float,
    "ph": float, "rainfall": float
  },
  "recommended_crop": "string",
  "created_at": "datetime (UTC)"
}
```

### `disease_predictions`
```json
{
  "_id": ObjectId,
  "user_id": "string",
  "image_filename": "string (stored in /uploads)",
  "disease_name": "string",
  "confidence": float,
  "cause": "string",
  "solution": "string",
  "created_at": "datetime (UTC)"
}
```

### `disease_solutions`
```json
{
  "_id": ObjectId,
  "disease_name": "string (unique, matches model class label)",
  "cause": "string",
  "solution": "string"
}
```

## 4. API / Page Flow

1. `GET/POST /register` → validates input → hashes password → inserts into `users` → redirect to `/login`.
2. `GET/POST /login` → finds user by username/email → `bcrypt.checkpw` → sets `session['user_id']`, `session['username']` → redirect to `/dashboard`.
3. `GET /logout` → `session.clear()` → redirect to `/login`.
4. `GET /dashboard` (login required) → fetch user doc, last 5 from `crop_predictions` & `disease_predictions`, total counts.
5. `GET/POST /predict-crop` (login required) → validate N,P,K,temperature,humidity,ph,rainfall → `predict_crop()` → insert into `crop_predictions` → render result.
6. `GET/POST /predict-disease` (login required) → validate image → save to `/uploads` → `predict_disease()` → lookup `disease_solutions` by `disease_name` → insert into `disease_predictions` → render result with cause/solution.
7. `GET /history` (login required) → all docs from `crop_predictions` and `disease_predictions` for the user, sorted by date desc.

## 5. Setup & Testing Steps

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your trained models
#    models/disease_model.h5
#    models/crop_model.pkl

# 4. Update config.py -> DISEASE_CLASSES to match your model's training label order exactly

# 5. Start MongoDB locally (mongod) - ensure it's running on localhost:27017

# 6. Run the app
python app.py
```

### Manual Test Checklist
1. Visit `http://localhost:5000/` → redirects to `/login`.
2. Register a new user → check `users` collection in MongoDB Compass for hashed password.
3. Try registering same username/email again → should show "already registered" error.
4. Login with correct/incorrect credentials → verify session set and error messages.
5. Go to `/predict-crop`, submit N,P,K,temperature,humidity,ph,rainfall → verify result shown and a new doc in `crop_predictions`.
6. Submit invalid (non-numeric / out-of-range pH) values → verify validation errors.
7. Go to `/predict-disease`, upload a leaf image (jpg/png) → verify disease name, confidence, cause, and solution shown; check `disease_predictions` collection and `/uploads` folder.
8. Upload a non-image file (e.g., .txt) → should be rejected.
9. Go to `/dashboard` → verify user info, recent items, and counts match DB.
10. Go to `/history` → verify all past predictions listed.
11. Logout → verify session cleared and protected routes redirect to `/login`.
