# [WIP] Number Recogition ML Model

This program is a tensorflow neural network which is  trained of from the MNIST dataset to recognise written numbers from 0 - 10. This is my first time project which uses neural networks so please leave any useful suggestions.

### Notice ⚠️ 
The program is trained from the MNIST dataset so if your numbers are drawn differently, the program *may* not recognize it.For best results, try draw the number in the middle and maybe you could slightly emphasize redeeming features of the number(e.g if you draw a weird 7, it could mistake it for a 4.) **OR** you could draw a bunch of numbers, save them (by settings **save_drawn_images** to `true` ) and train a new model on the **saved images**.

You could also tweak the amount of hidden layers,epochs and neurons.

## Installation
**NOTICE: This guide assumes you have [python](https://www.python.org/) installed.**
1) Download the repo (preferably using git)
```
git clone https://github.com/soakingdry/ml-number-reader
cd ml-number-reader
```
2) Install the program requirements
```
pip3 install -r requirements.txt
```
3) Open the `config.json` file and edit each field how you see fit - (guide listed below)

| Key | Value | Description |
| --- | --- | --- |
| `save_drawn_images` | bool(true/false) | Set to true if you want to save the images you draw to the "tmp" folder. This can be used to train the model so it recognizes your drawings better
| `visualize_tests` | bool(true/false) | Set to true if you want to see each number the model tries to guess when running a test

4) Run the program
```
python3 main.py
```


### Paint Controls
| Key |  Description | 
| --- | ---     
| `Left Click` | Draw/paint pixels
| `Right Click` | Eraser 
| `Enter` | Predict Number

## Other
**If you want to "recreate" the model, delete the "nn.model" file and rerun the program.**