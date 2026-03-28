# Tomato Leaf Disease Classifier (Real-World Images)

This is a **mini-project** that classifies **tomato leaf diseases** from **real-world (field) images**, not controlled lab images.

It uses the **Mendeley Data “Tomato leaf diseases” dataset (DOI: 10.17632/93h9p62kg4.1)**, which contains ~2600 images captured in natural field conditions with an iPhone 11.

The model is a **PyTorch CNN using transfer learning (ResNet18)** trained to classify images into **10 tomato leaf disease classes (plus healthy)**.  
A simple **Flask web app** lets you upload an image and see the predicted class.

---

## 1. Project structure

```text
mini-project/
  app.py                # Flask web app for inference
  train_model.py        # Training script (ResNet18 transfer learning)
  requirements.txt      # Python dependencies
  README.md             # This file
  models/
    tomato_resnet18.pth # Saved trained model (created after training)
  data/
    raw/                # Place original Mendeley images here (your choice of layout)
    dataset/
      train/
        CLASS_NAME_1/
        ...
      val/
        CLASS_NAME_1/
        ...
  templates/
    index.html          # Upload page
  static/
    styles.css          # Simple modern styling
```

> Note: `models/` and `data/` folders will be created by you / scripts; they are not committed by default.

---

## 2. Dataset (real-world, not controlled)

We use this dataset:

- **Name**: Tomato leaf diseases  
- **Provider**: Mendeley Data  
- **DOI**: `10.17632/93h9p62kg4.1`  
- **URL**: `https://data.mendeley.com/datasets/93h9p62kg4`  
- **Description** (summary):
  - ~2600 images of tomato leaves
  - Captured in **field conditions** (Khagan, Charabag, near Daffodil International University)
  - 10 disease types + healthy + small “other” class
  - Images taken using **iPhone 11** in natural lighting

### 2.1. Target classes

The description lists these labeled categories:

- Tomato Leaf Curl Virus  
- Spider Mites  
- Leaf Mold  
- Leaf Miner  
- Late Blight  
- Insect Damage  
- Healthy Leaves  
- Early Blight  
- Cercospora Leaf Mold  
- Bacterial Spot  
- (Optional) Misc/Other class

For this project we will use **10 classes** (you can choose which 10 to include).  
A reasonable mapping is:

1. `bacterial_spot`
2. `early_blight`
3. `late_blight`
4. `leaf_mold`
5. `leaf_miner`
6. `spider_mites`
7. `leaf_curl_virus`
8. `cercospora_leaf_mold`
9. `insect_damage`
10. `healthy`

You can adapt names to match whatever folder names you actually use.

---

## 3. Preparing the data

1. **Download the real-world dataset automatically**:

```bash
python download_dataset.py
```

This will download the Mendeley real-world field images into:

```text
data/raw/<class_name>/*.jpg
```

2. **Create train/val split automatically**:

```bash
python prepare_dataset.py
```

This creates:

```text
data/dataset/train/<class_name>/*.jpg
data/dataset/val/<class_name>/*.jpg
```

> If you want a faster demo download first, open `download_dataset.py` and set `limit_per_class` to e.g. `50`.

## 4. Setting up the environment

From inside `mini-project/`:

```bash
python -m venv .venv
.venv\Scripts\activate          # On Windows PowerShell
# source .venv/bin/activate     # On macOS / Linux

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Training the model

The training script:

- Uses **ResNet18** pre-trained on ImageNet
- Replaces the final layer to output **10 classes**
- Applies standard **data augmentation** (random flips, rotations, color jitter)
- Trains with **cross-entropy** and **Adam** optimizer
- Saves the model weights to `models/tomato_resnet18.pth`

Run:

```bash
python train_model.py
```

You can adjust:

- Number of epochs
- Learning rate
- Batch size
- Image size

in `train_model.py` if you want.

---

## 6. Running the web app

Once training finishes and `models/tomato_resnet18.pth` exists:

```bash
python app.py
```

Then open your browser at:

- `http://127.0.0.1:5000/`

Use the UI to:

1. Upload a tomato leaf image (JPG/PNG) from the **real-world dataset** (or your own field photo).
2. The backend:
   - Loads the trained ResNet18 model
   - Preprocesses the image the same way as in training
   - Runs a **softmax** and takes **argmax** to pick the **single best class**
3. The page shows:
   - The **predicted disease class**
   - The **probability** for that class (e.g. 0.87)

> There is **no threshold logic** like “if probability < 30% pick a different disease”. It always picks exactly **one** class using the model’s learned patterns.

---

## 7. Notes & ideas for improvement

- **Data balancing**: Some classes (e.g. Leaf Mold) have fewer images. You can:
  - Use stronger augmentation for small classes
  - Use class-weighted loss
- **More advanced architectures**:
  - Try EfficientNet, DenseNet, or ConvNeXt via `torchvision.models`
- **Better UI**:
  - Show top-3 predicted diseases and their probabilities
  - Display example images per class for comparison
- **Deployment**:
  - Wrap the Flask app in Docker
  - Deploy to a cloud service (Render, Railway, etc.)

This repo gives you a **complete simple project**: real-world dataset, proper training script, and a **one-click web demo** for tomato leaf disease classification.

