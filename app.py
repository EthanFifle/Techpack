from flask import Flask, request, render_template_string
import os
import requests
import shutil
import subprocess
import extract
import json

app = Flask(__name__)

# Ensure the upload and JSON directories exist
UPLOAD_DIRECTORY = "data/images"
KEYPOINTS_DIRECTORY = "data/keypoints"
OUTPUT_DIRECTORY = "output"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(KEYPOINTS_DIRECTORY, exist_ok=True)

HTML_TEMPLATE = '''
<!doctype html>
<html>
<head>
  <title>Upload new File</title>
</head>
<body>
  <h1>Upload new File</h1>
  <form method=post enctype=multipart/form-data>
    <label for="gender">Gender:</label>
    <select name="gender" id="gender" required>
      <option value="MALE">Male</option>
      <option value="FEMALE">Female</option>
      <option value="NEUTRAL">Neutral</option>
    </select>
    <br><br>
    <input type=file name=file required>
    <input type=submit value=Upload>
  </form>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files or 'gender' not in request.form:
            return 'File and gender are required'

        file = request.files['file']
        gender = request.form['gender']

        if file.filename == '':
            return 'No selected file'

        if file:
            # Clear the directories before saving new files
            clear_directory(UPLOAD_DIRECTORY)
            clear_directory(KEYPOINTS_DIRECTORY)
            clear_directory(OUTPUT_DIRECTORY)

            filename = file.filename
            filepath = os.path.join(UPLOAD_DIRECTORY, filename)
            response_filename = f"{os.path.splitext(filename)[0]}_keypoints.json"
            response_filepath = os.path.join(KEYPOINTS_DIRECTORY, response_filename)

            file.save(filepath)

            # Send the file to openpose-api
            response = extract_keypoints(filepath)
            with open(response_filepath, 'w') as f:
                f.write(response.text)

            result = call_smplifyx()
            if result.returncode != 0:
                # Handle the error case
                return f"Failed to run smplify-x command: {result.stderr}"

            pkl_path = get_pkl_file_path(f"output/results/{os.path.splitext(filename)[0]}")
            if pkl_path is None:
                return "No .pkl file found in the output/results directory."

            betas = extract_betas(pkl_path)
            if betas is None:
                return "Failed to extract betas from the .pkl file."

            measurements_response = get_measurements(gender, betas)
            if measurements_response .status_code != 200:
                return f'Failed to send data to port 5000. Status code: {measurements_response .status_code}'

            measurements_data = measurements_response.json()

            return f'All processes completed successfully, Measurements: {measurements_data}'

    return render_template_string(HTML_TEMPLATE)

def extract_keypoints(image_path):
    url = "http://localhost:8081/process_image"
    files = {'image': open(image_path, 'rb')}
    response = requests.post(url, files=files)
    return response

def get_measurements(gender, betas):
    url = "http://localhost:5000/upload"
    data = {
        'gender': gender,
        'betas': json.dumps(betas)  # Convert betas list to a JSON string
    }
    print(gender)
    print(json.dumps(betas))
    response = requests.post(url, data=data)
    return response
def clear_directory(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def call_smplifyx():
    cmd = [
        "python", "smplifyx/main.py",
        "--config", "cfg_files/fit_smpl.yaml",
        "--data_folder", "data",
        "--output_folder", "output",
        "--visualize", "False",
        "--model_folder", "models",
        "--vposer_ckpt", "vposer_v1_0"
    ]
    result = subprocess.run(cmd)
    return result

def get_pkl_file_path(directory):
    for file in os.listdir(directory):
        if file.endswith('.pkl'):
            return os.path.join(directory, file)
    return None

def extract_betas(pkl_path):
    betas_list = extract.extract_betas(pkl_path)
    return betas_list

if __name__ == '__main__':
    app.run(debug=True, port=3000)

