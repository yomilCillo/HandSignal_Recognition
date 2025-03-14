# HandSignal_Recognition


## 1. Software Preparation

### a) Download PyCharm Community Edition
PyCharm is an Integrated Development Environment (IDE) for Python. The Community Edition is free and suitable for beginners. You can download it from [PyCharm Downloads](https://www.jetbrains.com/pycharm/download/?section=windows).

### b) Install PyCharm
Once downloaded, run the installer and follow the on-screen instructions to install PyCharm on your PC or laptop.

### c) Download Python
Python is the programming language used for this project. Download the latest version from [Python Downloads](https://www.python.org/downloads/) and make sure to select the appropriate version for your operating system.

### d) Install Python
After downloading, run the Python installer and follow the installation prompts. **Make sure to check the box that says "Add Python to PATH"** during installation to easily access Python from the command line.

---

## 2. Setting up the Training Environment

### a) Prepare Teachable Machine
Teachable Machine is a web-based tool provided by Google for training machine learning models with image, sound, or pose data. Visit [Teachable Machine](https://teachablemachine.withgoogle.com/) to set up the environment for training the model with hand signal images.

---

## 3. Creating the PyCharm Project

### a) Open PyCharm
Launch PyCharm and create a new project by selecting **File > New Project**. Name the project and choose the location where you want to save it.

### b) Create Python Files
Within the project, create two Python files named `dataCollection.py` and `test.py`. These files will contain the code for collecting hand signal data and testing the trained model, respectively.

---

## 4. Training the Data

### a) Prepare Teachable Machine
Visit [Teachable Machine](https://teachablemachine.withgoogle.com/) to set up classes for each letter of the sign language alphabet.

### b) Upload Data
For each class (representing a sign language letter), upload the collected image data corresponding to that letter. Ensure that each class has a sufficient number of images for effective training.

### c) Start Training
Once all data is uploaded, click on the **"Train Model"** button to start the training process. This process involves training a machine learning model to recognize the hand signals corresponding to each letter.

### d) Export Model
After training is complete, click on the **"Export Model"** button and select the **TensorFlow-Keras** option to export the trained model. This model will be used for testing and validation.

---

## 5. Testing the Program

### a) Run `test.py`
Execute the `test.py` file in PyCharm to activate the camera and display real-time hand signal detection.
