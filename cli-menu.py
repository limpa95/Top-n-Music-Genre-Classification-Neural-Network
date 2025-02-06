from pyfiglet import Figlet
from termcolor import colored
from colorama import init as colorama_init
import os
from metrics.accuracy_metrics import display_accuracy

# Initialize colorama to allow colored terminal text in Windows OS.
colorama_init()


def banner():
    """Uses ASCII text to display a banner. Prints banner to console."""

    text = """                        Top-n
              Music Genre
          Classification
    Neural Network"""
    f = Figlet(font="Ogre")
    pyfiglet_text = f.renderText(text)
    colored_text = colored(pyfiglet_text, 'cyan')
    print(colored_text)


# Call banner function outside while loop to display title of program only once.
banner()

while True:
    print("1: Song classification instructions")
    print("2: Exit")

    choice = input("Please enter your choice: ")

    choice = choice.strip()

    if choice == "1":
        print("\nPlace your song into the 'single_input' folder. The format needs to be .wav.\n")

        current_dir = os.path.dirname(__file__)
        spectrogram_path = os.path.join(current_dir, 'single_output', 'png')

        if os.listdir(spectrogram_path):
            display_accuracy()
        else:
            print("\nNo file found. Try again\n")
    elif choice == "2":
        break
    else:
        print("Error: Not a valid choice. Please retry.")
