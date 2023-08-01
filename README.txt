*****************************************************



*****************************************************

To Execute SeeMore-InstanceSegmentationModel:
Go inside the folder SeeMore-InstanceSegmentationModel and do the following (Make sure to have torch 1.12 and timm):
1) Create a conda environment with python 3.8
2) Install all the packages as mentioned in requirements.txt using pip install -r requirements.txt
3) Then in the anaconda terminal just type: python seemore-voice.py (All figures from 6-10 in the report are generated from this code)

----------------------------------------

To execute SeeMore-Object_Detection - this is SSD model:
Go inside the folder SeeMore-Object_Detection and do: (Create conda env with python 3.7)
1) Install all the packages as mentioned in requirements.txt using pip install -r requirements.txt
2) Then go cd \models\research\object_detection
3) Execute in conda terminal: python webcam_blind_voice.py

If you want to run this on GPU change useGPU to 1 in the code.

----------------------------------------

To execute DeepLab model: (
Go inside SeeMore-DeepLab and make sure to have torch 1.10, opencv and python 3.8 with your env(you can use same env as used for SeeMore-InstanceSegmentationModel):
1) Execute: python midas_depth.py

If any dependencies are missing, please install them.

------------------------------------------

To execute Yolo model:(make sure to install ultralyics, opencv-python, pyttsx3)
Go inside the folder run:
1) python yolo_segmentation.py
2) python yolo_detection.py

If any dependencies are missing, please install them.


If there is any trouble while executing please contact any one of the team members.

--------------------------------------------

To execute ResNet50 (which is the vanilla implementation):
1) Run: python Resnet50.py


---------------------------------------------
