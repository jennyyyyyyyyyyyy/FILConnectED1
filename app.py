from flask import Flask, jsonify, request, redirect, session
from mediapipe_model_maker import gesture_recognizer
import os
import dropbox
import io
import requests
import tensorflow as tf
from dropbox.exceptions import ApiError

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'by6aaowh3zxrd5b'  # Change this to a secure key in production

# Dropbox App Key and Secret
APP_KEY = 'your_app_key'
APP_SECRET = '5lvz2hzydmsssee'

# OAuth Redirect URI
REDIRECT_URI = 'https://filconnected1.onrender.com/oauth/callback'

# Dropbox API Access Token Placeholder (will be set after OAuth)
DROPBOX_ACCESS_TOKEN = None

# Dataset and Export Model Paths
DATASET_PATH = '/gesture_dataset'
EXPORT_MODEL_PATH = '/exported_model/gesture_recognizer.task'


# Function to get Dropbox OAuth flow
def get_dropbox_auth_flow():
    return dropbox.oauth.DropboxOAuth2Flow(
        APP_KEY,
        APP_SECRET,
        REDIRECT_URI,
        session,
        "dropbox-auth-csrf-token"
    )


# Function to get Dropbox client
def get_dropbox_client():
    if not DROPBOX_ACCESS_TOKEN:
        raise Exception("User is not authenticated. Please log in.")
    return dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)


# Function to get dataset from Dropbox
def get_dataset_from_dropbox():
    print("Accessing dataset from Dropbox...")
    try:
        dbx = get_dropbox_client()
        folder_metadata = dbx.files_list_folder(DATASET_PATH)
        dataset = {}

        for entry in folder_metadata.entries:
            if isinstance(entry, dropbox.files.FileMetadata):
                file_path = entry.path_display
                metadata, res = dbx.files_download(file_path)
                dataset[file_path] = io.BytesIO(res.content)

        print("Dataset ready for processing.")
        return dataset
    except ApiError as e:
        raise Exception(f"Error accessing dataset from Dropbox: {e}")


# Function to save the exported model to Dropbox
def save_model_to_dropbox(local_model_path):
    print("Uploading trained model to Dropbox...")
    try:
        dbx = get_dropbox_client()
        with open(local_model_path, 'rb') as f:
            dbx.files_upload(f.read(), EXPORT_MODEL_PATH, mode=dropbox.files.WriteMode.overwrite)
        print("Model uploaded successfully.")
    except Exception as e:
        raise Exception(f"Error uploading model to Dropbox: {e}")


# Route: Redirect to Dropbox Login
@app.route('/oauth/start')
def oauth_start():
    auth_flow = get_dropbox_auth_flow()
    authorize_url = auth_flow.start()
    return redirect(authorize_url)


# Route: Handle OAuth Callback
@app.route('/oauth/callback')
def oauth_callback():
    try:
        auth_flow = get_dropbox_auth_flow()
        access_token, user_id, url_state = auth_flow.finish(request.args)
        global DROPBOX_ACCESS_TOKEN
        DROPBOX_ACCESS_TOKEN = access_token
        session['access_token'] = access_token
        return jsonify({"message": "Authentication successful!", "access_token": access_token})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Route: Example API Endpoint with Authentication
@app.route('/list_files')
def list_files():
    try:
        dbx = get_dropbox_client()
        files = dbx.files_list_folder('').entries
        file_names = [file.name for file in files if isinstance(file, dropbox.files.FileMetadata)]
        return jsonify({"files": file_names})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Route: Train Model
@app.route('/train', methods=['GET'])
def train_model():
    try:
        # Access dataset directly from Dropbox
        dataset = get_dataset_from_dropbox()

        # Load and preprocess dataset
        labels = [os.path.basename(file_path) for file_path in dataset.keys()]
        print(f"Labels: {labels}")

        # Assuming you can modify gesture_recognizer to work with in-memory data
        data = gesture_recognizer.Dataset.from_memory(
            dataset=dataset,
            hparams=gesture_recognizer.HandDataPreprocessingParams()
        )

        train_data, rest_data = data.split(0.8)
        validation_data, test_data = rest_data.split(0.5)

        # Train the model
        hparams = gesture_recognizer.HParams(export_dir="exported_model")
        options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
        model = gesture_recognizer.GestureRecognizer.create(
            train_data=train_data,
            validation_data=validation_data,
            options=options
        )

        # Evaluate the model
        loss, acc = model.evaluate(test_data, batch_size=1)
        print(f"Test loss: {loss}, Test accuracy: {acc}")

        # Export the model
        model.export_model()
        local_task_file = "exported_model/gesture_recognizer.task"

        # Upload exported model to Dropbox
        if os.path.exists(local_task_file):
            save_model_to_dropbox(local_task_file)
            return jsonify({
                "message": "Model trained and exported successfully!",
                "test_loss": loss,
                "test_accuracy": acc,
                "dropbox_path": EXPORT_MODEL_PATH
            })
        else:
            return jsonify({"error": "Model training completed but .task file was not found!"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Route: Upload File to Dropbox
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request."}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected for uploading."}), 400

        dropbox_path = f"{DATASET_PATH}/{file.filename}"
        dbx = get_dropbox_client()
        dbx.files_upload(file.read(), dropbox_path, mode=dropbox.files.WriteMode.overwrite)

        return jsonify({"message": "File uploaded successfully!", "dropbox_path": dropbox_path})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Route: Home
@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Gesture Recognition Model API!"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
