 Colon Cancer Image Classification

 Overview

This project involves the development of a convolutional neural network (CNN) to classify histopathological images of colon cancer. The model is trained on a subset of images and uses data augmentation techniques to improve performance with the help of ChatGPT and Copilot .

 My Project Team:
Muhammet Berke Ağaya
Ayça Sena Kayhan
Aysu Öztürk
Afra Nevim Özkılıç
Efe Gerkin

 Dataset

The dataset used in this project consists of histopathological images of colon cancer, specifically adenocarcinoma (ACA) and normal (N). These images are sourced from the [Lung and Colon Cancer Histopathological Images dataset](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images). You can access the dataset [here](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images).

 Preprocessing

- Rescaling: Images are rescaled to normalize pixel values to the range [0, 1].
- Data Augmentation: The `Image Data Generator` class is used for augmentation, including validation splitting.
- Subset Selection: Only the first 200 images from each class are used for simplicity.

 Model Architecture

The CNN model consists of the following layers:

1. Conv2D: 32 filters, 3x3 kernel, ReLU activation.
2. MaxPooling2D: 2x2 pooling.
3. Conv2D: 64 filters, 3x3 kernel, ReLU activation.
4. MaxPooling2D: 2x2 pooling.
5. Flatten: Converts 2D feature maps into a 1D vector.
6. Dense: Fully connected layer with 128 neurons and ReLU activation.
7. Dropout: Dropout rate of 0.5 to reduce overfitting.
8. Dense: Output layer with 1 neuron and sigmoid activation for binary classification.

 Training

- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Metrics: Accuracy
- Epochs: 10
- Batch Size: 32

The data is split into training (90%) and validation (10%) sets.

 Results

The model achieved a validation accuracy of approximately 80%, as shown in the training log. Below is an excerpt from the training output:


Epoch 10/10
2s 61ms/step - accuracy: 0.8005 - loss: 0.3717 - val accuracy: 0.8000 - val loss: 0.5161


The model's accuracy and loss are plotted for both training and validation datasets to visualize performance.

 Charts :

https://drive.google.com/drive/folders/1bbLx2GhsZ4J0ZgWAckR16FqZrVMJvHkO?usp=sharing

 Accuracy

The accuracy plot shows the learning progress over epochs for both training and validation datasets.

The loss plot visualizes how the error decreases during training and validation.



 Additional Context

This project was developed during a competition as part of the “Sağlıkta Yapay Zeka: Öğrenmekten Üretmeye” training event. Guided by Dr. Yasin Durusoy, participants from diverse disciplines, including computer and biomedical engineering, dentistry, and medical students, engaged in a hands-on process to learn the fundamentals of AI model development. The event involved both theoretical and practical sessions, culminating in a friendly competition where participants worked in multidisciplinary groups to address significant healthcare problems using AI models.

This code, along with the accompanying presentation, secured 2nd place in the competition. The event highlighted teamwork, innovation, and the application of AI in healthcare, and included awards for the top-performing teams.

 Usage

1. Clone the repository.
2. Place the dataset in the specified directory structure.
3. Run the script to train the model.

 Dependencies

- Python 3.7+
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- scikit-learn

Install dependencies with:


pip install tensorflow numpy pandas matplotlib scikit-learn


 If you have questions about project, you can contact me via my linkedin account : https://www.linkedin.com/in/muhammet-berke-a%C4%9Faya-63b4692a2/


