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


def check_files(path):
    """Checks if files exist and calls display_accuracy function to show top-n graph.
    Returns false to indicate folder directory is not empty. Otherwise returns true.
    """

    if len(os.listdir(path)) > 0:
        files_list = os.listdir(path)

        # Call metrics function to display accuracy.
        display_accuracy(files_list)

        # Remove png files from directory to prep for new file.
        for file in files_list:
            os.remove(os.path.join(path, file))

        return False
    return True


def menu():
    """Uses while loop to run CLI menu. Provides instructions to the user and displays top-n graph."""

    # Call banner function outside while loop to display title of program only once.
    banner()

    while True:
        print("1: Song classification")
        print("2: Exit")

        choice = input("Please enter your choice: ")

        choice = choice.strip()

        if choice == "1":

            # Keep track of file deletion and error message
            empty = True

            print("\nPlace your song into the 'single_input' folder under 'audio_processing.' "
                  "The format needs to be .wav.\n")

            current_dir = os.path.dirname(__file__)
            png_spectrogram_path = os.path.join(current_dir, 'audio_processing', 'single_output', 'png')
            npy_spectrogram_path = os.path.join(current_dir, 'audio_processing', 'single_output', 'npy')

            # Pause program to wait for user to place music file and continue after pressing enter.
            input("Once the files have finished converting, press enter to continue.\n")

            if os.path.exists(png_spectrogram_path):
                empty = check_files(png_spectrogram_path)

            if os.path.exists(npy_spectrogram_path):
                empty = check_files(png_spectrogram_path)

            if empty is True:
                print("No files found. Try again\n")

        elif choice == "2":
            break
        else:
            print("Error: Not a valid choice. Please retry.\n")


if __name__ == '__main__':
    menu()
