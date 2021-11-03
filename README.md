# NYCU_VRDL_Homework1
This is a Homework in VRDL class at NYCU , building a classifier of bird photo with deep learning

```requirements.txt```  need to install these librarys

```Train.py``` Training code 

```Inference.py``` Predicting code
## Requirements
To install requirements:
```pip install -r requirements.txt```

My Python version==3.7.10
## Training
```Train.py``` is Training code with Python

```python Train.py --folder [train_folder_path] --label [train_label_path] ```

will save model in ```./model``` folder  and save ```result.png``` 
## Evaluation
```Inference.py``` is Predicting code with Python

```python inference.py --folder [test_folder_path] --labelmap [path:classes.txt] --order [path:testing_img_order.txt]' --model [model_path]```

will save ```answer.txt```  as predicting result
## Pre-trained model
[Pretrained model](https://drive.google.com/file/d/181dg8oS8tRkA4JDfZkPHasuwF9qn0dS-/view?usp=sharing) trained on Bird dataset (EfficientNet-b7)

I used EfficientNet-b7,b5 model of [EfficientNet PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch) open source 
## Results
my model's accuracy:
|model name|Accuracy|
|---|--|
|EfficientNet-b7|0.73492|
|EfficientNet-b5|0.677217|
|ResNet50|0.610617|
|denseNet161|0.607979|
