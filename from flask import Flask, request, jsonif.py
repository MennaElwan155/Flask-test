from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('cnn_model.h5')

# Initialize the Flask app
app = Flask(__name__)

# Define the route 'menna'
@app.route('/menna', methods=['POST'])
def menna():
    # Get the JSON data from the request
    data = request.get_json(force=True)
    # Convert the data into a NumPy array
    image = np.array(data['image'])
    # Ensure the image has the correct shape
    image = image.reshape((1, 28, 28, 1))
    # Make a prediction
    prediction = model.predict(image)
    # Get the class with the highest probability
    predicted_class = np.argmax(prediction, axis=1)
    # Return the result as a JSON response
    return jsonify({'predicted_class': int(predicted_class[0])})

if __name__ == '__main__':
    app.run(debug=True)
