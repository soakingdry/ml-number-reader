#  AI Number Reader

This program uses a neural network made with tensorflow which is  trained of from the MNIST dataset to recognize digits you've drawn that are from 0-9.
<img src="canvas_eg.png">

### Notice ⚠️ 
The program is trained from the MNIST dataset so if your numbers are drawn differently, the program *may* not recognize it.For best results, try draw the number in the middle and maybe you could slightly emphasize redeeming features of the number(e.g if you draw a weird 7, it could mistake it for a 4.) **OR** you could draw a bunch of numbers, save them (by settings **save_drawn_images** to `true` ) and train a new model on the **saved images**.


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
**If you want to recreate the model, delete the "cnn_mnist_model.keras" file and rerun the program.**

