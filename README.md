# Simple Digit Recognition (WIP)

An application which uses a CNN trained on the MNIST dataset to identify handwritten images drawn from 0-9

## Usage

### Training a model
The program will not work if you have not trained a model.
```commandline
python train.py  --name any_model_name_here 
```

Hyperparameters such as `--epochs`, and `--lr` and `--hidden_units` are optional and can be adjusted based on model performance. 
You can view the source of `train.py` if you want more insight on these parameters

### Running the program

Once you have trained the model, you can run the canvas program
```commandline
python main.py --model model_name_here
```

### Controls
| Key |  Description | 
| --- | ---     
| `Left Click` | Draw/paint pixels
| `Right Click` | Eraser 
| `Enter` | Predict Number


## TODO:
- Implement early stopping

