import colorama
import json 
from colorama import Style,Fore

from utils import log 
from utils.visuals.canvas import Canvas
from utils.model.test_model import ModelTesting

def main():
    with open("config.json") as file:
        config = json.load(file)
        
    log.info(config,prefix="config")

    pRunTest = input(f"{Style.BRIGHT}Do you want to run a test on the model (using the MNIST dataset)? [y/n] ")
    if pRunTest.lower() == "y":
        
        visual = config["visualize_tests"]

        model = ModelTesting(visualize=visual)
        model.test()
 
    pRunCanvas = input(f"{Style.BRIGHT}Do you want to load a paint program where you can draw a number and the program will attempt to predict the number? [y/n] ")
    if pRunCanvas.lower() == "y":
        log.warn("The program is trained from the MNIST dataset so if your numbers are drawn differently, the program *may* not recognize it.For best results, try draw the number in the middle and maybe you could slightly emphasize redeeming features of the number(e.g if you draw a weird 7, it could mistake it for a 4, emphasize the slant on the tail of the 7 and the placing of the lines.)",prefix="!")        
        log.info("Controls:\nLeft Click - Draw\nRight Click - Rubber\nEnter - Predict","!")

        PaintProgram = Canvas(title="Draw a number (0-9)")
        PaintProgram.display()

if __name__ == "__main__":
    main()