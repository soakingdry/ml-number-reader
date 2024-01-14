import argparse
from utils.visuals import canvas


def main():
    parser = argparse.ArgumentParser(description="Canvas app")
    parser.add_argument("--model", type=str, help="The name of the model you want to use")
    args = parser.parse_args()
    canvas_app = canvas.Canvas(model_name=args.model, title="Draw a number (0-9)")
    canvas_app.display()


if __name__ == "__main__":
    main()
