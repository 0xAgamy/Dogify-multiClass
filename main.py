import tf_keras as tfk
import tensorflow_hub as hub
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io


MODEL_PATH = "Models/20241019-165149-1000-images-mobilenetv2-Adam.h5"
# Custom object dictionary for loading TensorFlow Hub layer
custom_objects = {'KerasLayer': hub.KerasLayer}

# Load the fine-tuned MobileNetV2 model
model = tfk.models.load_model(MODEL_PATH, custom_objects=custom_objects)

# Load class names or labels (update based on your dataset)
class_names = ['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale',
       'american_staffordshire_terrier', 'appenzeller',
       'australian_terrier', 'basenji', 'basset', 'beagle',
       'bedlington_terrier', 'bernese_mountain_dog',
       'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound',
       'bluetick', 'border_collie', 'border_terrier', 'borzoi',
       'boston_bull', 'bouvier_des_flandres', 'boxer',
       'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff',
       'cairn', 'cardigan', 'chesapeake_bay_retriever', 'chihuahua',
       'chow', 'clumber', 'cocker_spaniel', 'collie',
       'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo',
       'doberman', 'english_foxhound', 'english_setter',
       'english_springer', 'entlebucher', 'eskimo_dog',
       'flat-coated_retriever', 'french_bulldog', 'german_shepherd',
       'german_short-haired_pointer', 'giant_schnauzer',
       'golden_retriever', 'gordon_setter', 'great_dane',
       'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael',
       'ibizan_hound', 'irish_setter', 'irish_terrier',
       'irish_water_spaniel', 'irish_wolfhound', 'italian_greyhound',
       'japanese_spaniel', 'keeshond', 'kelpie', 'kerry_blue_terrier',
       'komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier',
       'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog',
       'mexican_hairless', 'miniature_pinscher', 'miniature_poodle',
       'miniature_schnauzer', 'newfoundland', 'norfolk_terrier',
       'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog',
       'otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian',
       'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler',
       'saint_bernard', 'saluki', 'samoyed', 'schipperke',
       'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier',
       'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier',
       'soft-coated_wheaten_terrier', 'staffordshire_bullterrier',
       'standard_poodle', 'standard_schnauzer', 'sussex_spaniel',
       'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 'toy_terrier',
       'vizsla', 'walker_hound', 'weimaraner', 'welsh_springer_spaniel',
       'west_highland_white_terrier', 'whippet',
       'wire-haired_fox_terrier', 'yorkshire_terrier'] # Modify this with your class labels


def preprocess_image(image_bytes):
    """
    Preprocess the image to be in the right format for MobileNetV2.
    - Resizes to 224x224
    - Scales pixel values to [0, 1]
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')  # Convert to RGB
    img = img.resize((224, 224))  # Resize image to 224x224
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array
app = Flask(__name__)


@app.route('/')
def home():
    return "Image Classification API is running."

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to handle image classification.
    Accepts an image, preprocesses it, and returns the top 3 predictions.
    """
    try:
        # Check if an image is sent in the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        # Get the image from the request
        image_file = request.files['image'].read()

        # Preprocess the image
        input_data = preprocess_image(image_file)

        # Make predictions
        predictions = model.predict(input_data)[0]  # Get first item from batch

        # Get top 3 predictions (sorted by probability)
        top_3_indices = np.argsort(predictions)[-3:][::-1]  # Get indices of top 3 predictions
        top_3_predictions = [(class_names[i], float(predictions[i])) for i in top_3_indices]

        # Return the top 3 predictions as JSON
        return jsonify({
            'predictions': top_3_predictions
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
