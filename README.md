# Sign Language Letter Classification with a CNN from Scratch

**Author: Eng. Noureldin Bassem Mohamed**

A single letter American Sign Language classifier: give it a photo of a hand shape and it returns
one of 24 letters (the alphabet without J and Z, which both need movement). The CNN is designed
and trained from scratch, no pretrained backbone.

## Files

| File | What it is |
| --- | --- |
| `sign-language-cnn.ipynb` | The full notebook: EDA, preprocessing, architecture and its reasoning, training, evaluation, write-up. All cells run top to bottom. |
| `app.py` | Streamlit app, loads the trained model and returns a real prediction for an uploaded photo. |
| `artifacts/asl_cnn.keras` | The trained model saved by the notebook. |
| `artifacts/class_names.json` | The 24 letter names and the input size, read by the app. |
| `artifacts/demo_sign.png` | A test set image written out as a PNG, handy for a quick app check. |
| `requirements.txt` | Dependencies. |
| `../app_demo.mp4` | 81 second screen recording of the app running, three photos uploaded and predicted live. |

## Dataset

[Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist), 27,455
training rows and 7,172 test rows, each row a 28x28 grayscale image flattened into 784 pixel
columns. The notebook downloads it with `kagglehub`, nothing needs to be placed by hand.

## The model

Three convolutional stages of 3x3 filters (32 → 64 → 128) with batch normalization after every
convolution, two max poolings (28 → 14 → 7), dropout growing from 0.25 to 0.5, a Dense(256) head
and a 24 way softmax. Trained with Adam at 1e-3, early stopping on validation accuracy and
`ReduceLROnPlateau`. Light augmentation only (small rotation, zoom and translation, no horizontal
flips, since mirroring a hand sign changes its meaning). The test CSV is held out completely and
used once, at the end.

## Running the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload a photo cropped close to the hand. The app shows the 28x28 image the model actually sees,
the predicted letter with its confidence, and the top three candidates. Real phone photos are
harder than the dataset images, which are tight crops that were already filtered and contrast
adjusted, so the sidebar has an invert toggle for photos that are dark hand on a light background.
