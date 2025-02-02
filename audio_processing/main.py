import audio_processing as ap
import time


"""
This is the main file for converting audio files to input into the AI Model. 
Run this file and place audio to be converted into the single_input directory
Audio will be converted and placed in the single_output directory. 
"""


data_folder_path = "single_input"
run = True
printed_waiting = False

while (run):

    try:

        files_list = ap.check_single_input_directory(
            ap.EmptyDirectoryError, data_folder_path)

        if files_list:
            files_list.sort()
            for audio_file in files_list:
                ap.process_single_audio_file(
                    audio_file, ap.convert_single_file_mel_spectrogam)

                print("File converted")

                ap.remove_converted_audio_file(data_folder_path, audio_file)
            printed_waiting = False

        else:
            if printed_waiting is False:
                # os.system('clear')
                print(f"Waiting for file input to {
                      data_folder_path} directory")
                print("To exit the program press 'ctrl+c'")
                printed_waiting = True
            time.sleep(5)

    except KeyboardInterrupt:
        print("Exiting the program.")
        run = False
