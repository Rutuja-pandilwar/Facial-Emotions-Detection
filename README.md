# Facial Emotion Detection 🎭

A simple Python project that detects the **emotion on a human face** from an image using the
[`FER`](https://pypi.org/project/fer/) library (CNN-based) and **OpenCV**, then draws a
bounding box around the face labeled with the detected emotion.

---

## 📌 What It Does

1. Loads an image.
2. Detects the face using **MTCNN** (Multi-task Cascaded Convolutional Network).
3. Predicts probability scores for **7 emotions**: `angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, `neutral`.
4. Draws a green bounding box around the detected face.
5. Labels the box with the **dominant (highest-confidence) emotion**.
6. Saves and displays the final annotated image.

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `opencv-python (cv2)` | Image reading, drawing rectangles/text, saving output |
| `fer` | Pre-trained CNN model for facial expression recognition |
| `matplotlib` | Displaying the final image inline |
| `numpy` | Numerical operations |

---

## 📂 Project Structure

```
facial-emotion-detection/
├── Facial_Emotion_Detection.ipynb   # Main notebook with step-by-step code
├── README.md                        # This file
└── facial_emotion.jpg               # Output image (generated after running)
```

---

## ⚙️ Installation

```bash
pip install opencv-python fer matplotlib numpy
```

> `fer` internally depends on TensorFlow/Keras and `mtcnn`, which will be installed
> automatically.

---

## 🚀 How to Run

1. Open `Facial_Emotion_Detection.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
2. Update the `imgloc` variable to point to your image file:
   ```python
   imgloc = '/content/your_image.jpeg'
   ```
3. Run all cells in order.
4. The output image (`facial_emotion.jpg`) with the bounding box + emotion label will be
   saved in your working directory and displayed in the notebook.

---

## 🧠 How It Works (High Level)

- **Face Detection:** `FER(mtcnn=True)` uses MTCNN to locate the face and return its
  bounding box `[x, y, width, height]`.
- **Emotion Classification:** A CNN trained on labeled facial expression datasets (like
  FER2013) outputs a probability score (0–1) for each of the 7 emotion classes.
- **Top Emotion:** `emo_detector.top_emotion(image)` returns the single most likely emotion
  and its confidence score.
- **Visualization:** OpenCV draws the bounding box (`cv2.rectangle`) and label
  (`cv2.putText`) directly onto the image array.

---

## ⚠️ Known Limitations / Notes

- This code processes only the **first detected face** (`captured_emotions[0]`). For
  multiple faces, you'd loop over the full `captured_emotions` list.
- OpenCV uses **BGR** color order, while matplotlib expects **RGB**. This code works around
  it by saving and re-reading the image; for a direct fix use
  `cv2.cvtColor(image, cv2.COLOR_BGR2RGB)`.
- No error handling for the case where **no face is detected** in the image — add a check
  like `if len(captured_emotions) == 0: ...` for production use.

---

## 🔮 Possible Improvements

- Loop through all detected faces, not just the first.
- Add webcam / real-time video support (`cv2.VideoCapture`).
- Add a confidence threshold before labeling an emotion.
- Wrap into a reusable function or small Streamlit/Gradio app for demo purposes.

---

## 📄 License

This project is for educational/demo purposes.
