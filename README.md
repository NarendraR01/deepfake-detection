# Deep fake detection Django Application
## Requirements:

**Note :** Nvidia GPU is mandatory to run the application.
- CUDA version >= 10.0 for GPU
- GPU Compute Capability > 3.0 

TML

<b>Note:</b> Before running the project make sure you have created directories namely <strong>models, uploaded_images, uploaded_videos</strong> in the project root and that you have proper permissions to access them.
# Running application on Docker
#### Step 1: Install docker desktop and start the Docker daemon

#### Step 2: Run the deepfake detection docker docker image
```
docker run --rm --gpus all -v static_volume:/home/app/staticfiles/ -v media_volume:/app/uploaded_videos/ --name=deepfakeapplication abhijitjadhav1998/deefake-detection-20framemodel
```
#### Step 3: Run the Ngnix reverse proxy server docker image
```
docker run -p 80:80 --volumes-from deepfakeapplication -v static_volume:/home/app/staticfiles/ -v media_volume:/app/uploaded_videos/ abhijitjadhav1998/deepfake-nginx-proxyserver
```
#### Step 4: All set now launch up your application at [http://localhost:80](http://localhost:80)

