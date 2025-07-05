# OCR-Lanna

This project provides optical character recognition (OCR) utilities for the Lanna script. A convolutional neural network (CNN) is used to classify characters and additional logic stitches them into words. A small Flask web application is included for interactive testing.

## Requirements

Install the dependencies with `pip`:

```bash
pip install -r requirements.txt
```

Python 3.10 or later is recommended.

## Usage

### Web Interface

Running `API.py` starts a Flask server and opens a local HTML page. Upload an image containing Lanna text and the application will return the recognized characters with similarity scores.

```bash
python API.py
```

### Command-Line Prediction

`Predict.py` performs OCR on a single sample image. By default it reads `image_for_test/18.jpg` and displays the detected characters.

```bash
python Predict.py
```

### Training

`Train_model.py` builds and trains the CNN using images located in `dataset/model/train` and `dataset/model/test`. The resulting model is saved as `Model/OCR-lanna.h5`.

```bash
python Train_model.py
```

### Template Matching Example

`Predict_matchTemplate.py` demonstrates prediction together with template matching to report similarity percentages for each detected character.

## Data

Sample images for inference are located in `image_for_test`. Training data for each character class is stored under `dataset/model`. A short manual describing the project is provided in `manual_OCR_lanna.pdf`.

## License

No license information is supplied in this repository.
