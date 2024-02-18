import base64
import requests
import time

def image_to_base64_str(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode('utf-8')

source_paths = ['./images/hou-1.JPG', './images/hou-2.JPG']
target_path = './images/test-2.mp4'

sources = []
for source_path in source_paths:
    sources.append(image_to_base64_str(source_path))
target = image_to_base64_str(target_path)


params = {
    'sources': sources,
    'target': target,
}

url = 'http://0.0.0.0:8000/'
response = requests.post(url, json=params)

print("Status Code:", response.status_code)
print("Response Body:", response.text)

if response.status_code == 200:
    output_data = base64.b64decode(response.json()['output'])
    with open(f'output/{int(time.time())}.jpg', 'wb') as f:
        f.write(output_data)
else:
    print("Error: The request did not succeed.")
