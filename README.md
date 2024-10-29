# Lung Cancer Detection

A machine learning model that detects and classifies lung cancer based on histopathological images.

Image...

## About

This machine learning model is trained on the [Lung Cancer (Histopathological Images) Dataset](https://www.kaggle.com/datasets/rm1000/lung-cancer-histopathological-images) with the goal of detecting and classifying lung cancer based on histopathological images. It's built completely from scratch with PyTorch and uses other libraries like pandas for handling the dataset, Matplotlib for plotting the training results and PrettyTable for printing the results to the console in table format.

## Dataset

The dataset is made up of 15,000 images equally divided into the 3 categories "benign lung tissue", "lung adenocarcinoma" and "squamous cell carcinoma". All images are 768x768 pixels in size, de-identified, HIPAA compliant and validated by medical experts.

Benign lung tissue refers to non-cancerous tissues or growths in the lung that do not invade surrounding tissues or spread to other parts of the body.
Lung adenocarcinoma is a type of non-small cell lung cancer (NSCLC) that originates in the glandular cells of the lung.
Squamous cell carcinoma (SCC) of the lung is another type of non-small cell lung cancer that arises from squamous cells, which are flat cells that line the airways of the lungs.

In summary, lung adenocarcinoma and squamous cell carcinoma are both types of lung cancer, with distinct characteristics and risk factors. Benign lung tissue refers to non-cancerous growths that do not pose a risk of metastasis.

## Results

The neural network was trained for 40 epochs with a learning rate of 0.001 using Adam as the optimizer and CrossEntropyLoss as the criterion. Training was done on 75% of the dataset while the remaining 25% were used to validate the results.

```
+-------+--------------+
| Epoch |     Loss     |
+-------+--------------+
|   1   | 0.4028602839 |
|   2   | 0.2948895395 |
|   4   | 0.2364200205 |
|   6   | 0.0671709329 |
|   8   | 0.0111341095 |
|   10  | 0.0002070912 |
|   12  | 0.0509917028 |
|   14  | 0.0000172644 |
|   16  | 0.0189744309 |
|   18  | 0.0000722565 |
|   20  | 0.0003521162 |
|   22  | 0.0002417870 |
|   24  | 0.0000676674 |
|   26  | 0.0000017285 |
|   28  | 0.0000047948 |
|   30  | 0.0000077815 |
|   32  | 0.0000057352 |
|   34  | 0.0000017881 |
|   36  | 0.0000040729 |
|   38  | 0.0000010000 |
|   40  | 0.0000001854 |
+-------+--------------+
```

Image...

The model achieved an accuracy of 95.71% on the testing set.

```
Test Accuracy: 95.71% (3589/3750)
```

## License

This software is licensed under the [MIT license](LICENSE).
