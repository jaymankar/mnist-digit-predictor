# Handwritten Digit Recognizer — End to End 🔢

A full end-to-end ML web app — draw a digit, get a prediction.  
CNN trained from scratch on MNIST · served with FastAPI · deployed live.

---

## 🌐 live demo

**[digit-predictor-jay.onrender.com](https://digit-predictor-jay.onrender.com)**

> ⚠️ first load takes 20-40s (free tier cold start)  
> if suspended, first load may take ~6 min — just wait, it'll come back

---

## results 📊

| Model | Dataset | Accuracy | Epochs |
|-------|---------|----------|--------|
| CNN (from scratch) | MNIST | **97%** | 50 |

---

## architecture 🧠

```
Input (1, 28, 28)
    │
    ▼
Conv2d(1 → 32, kernel=3)  →  ReLU  →  MaxPool2d(2x2)
    │
    ▼
Conv2d(32 → 64, kernel=3)  →  ReLU  →  MaxPool2d(2x2)
    │
    ▼
Flatten()
    │
    ▼
Linear(6455 → 128)  →  ReLU
    │
    ▼
Linear(128 → 10)
    │
    ▼
Output (10 classes: 0-9)
```

---

## how it works ⚙️

```
user draws digit on canvas (frontend)
        ↓
image sent as base64 to FastAPI backend
        ↓
preprocessing pipeline cleans + resizes image
        ↓
CNN model runs inference
        ↓
prediction + probabilities returned to frontend
```

---

## stack 🛠️

```
Backend   →  FastAPI · PyTorch · Python
Frontend  →  HTML Canvas · Vanilla JS
Model     →  CNN trained from scratch on MNIST
Serving   →  FastAPI REST API (/predict)
Deployed  →  Render (free tier)
```

---

## project structure 📁

```
mnist-end-to-end/
│
├── model/
│   ├── model.py             # CNN architecture
│   └── model.pth            # trained weights
│
├── Image_preprocessing/
│   └── preprocessing.py     # base64 → tensor pipeline
│
├── templates/
│   └── index.html           # draw canvas frontend
│
└── main.py                  # FastAPI app + routes
```

---

## API 📬

**GET** `/` — serves the drawing canvas frontend

**POST** `/predict`
```json
// request
{ "image": "<base64 encoded image>" }

// response
{
  "prediction": 7,
  "probabilities": [0.001, 0.002, ..., 0.95, ...]
}
```

---

## run it locally 🚀

```bash
git clone https://github.com/jaymankar/<repo-name>
cd <repo-name>

pip install fastapi uvicorn torch torchvision pillow

uvicorn main:app --reload
```

open `http://localhost:8000` → draw a digit → see prediction

---

## what I learned building this 🧠

- How to serve a PyTorch model as a REST API with FastAPI
- How to handle base64 image encoding between frontend and backend
- Building a preprocessing pipeline that matches training distribution
- Why CORS middleware matters for frontend-backend communication
- How to deploy a FastAPI app on Render
- End-to-end thinking — not just training a model, but shipping it live

---

*my first end-to-end ML project — from raw pixels to a live deployed prediction* 🔥  
*built at 18, no degree, just curiosity* 🙂
