# Banner function to make a title for the CLI menu was adapted from a tutorial by
# Naufal Ardhani at
# https://naufalardhani.medium.com/how-to-create-ascii-text-banner-for-command-line-project-85e75dc02b07
def banner():
    """Uses ASCII text to display a banner. Prints banner to console."""

    cyan = "\033[36m"
    text = f"""
                                        {cyan} _____                                                                   
                                        /__   \___  _ __        _ __                                             
                                          / /\/ _ \| '_ \ _____| '_ \                                            
                                         / / | (_) | |_) |_____| | | |                                           
                                         \/   \___/| .__/      |_| |_|                                           
                                                   |_|                                                           
                  _          ___                        ___ _               _  __ _           _   _              
  /\/\  _   _ ___(_) ___    / _ \___ _ __  _ __ ___    / __\ | __ _ ___ ___(_)/ _(_) ___ __ _| |_(_) ___  _ __   
 /    \| | | / __| |/ __|  / /_\/ _ \ '_ \| '__/ _ \  / /  | |/ _` / __/ __| | |_| |/ __/ _` | __| |/ _ \| '_ \  
/ /\/\ \ |_| \__ \ | (__  / /_\\  __/ | | | | |  __/ / /___| | (_| \__ \__ \ |  _| | (_| (_| | |_| | (_) | | | | 
\/    \/\__,_|___/_|\___| \____/\___|_| |_|_|  \___| \____/|_|\__,_|___/___/_|_| |_|\___\__,_|\__|_|\___/|_| |_| 
                                                                                                                 
                         __                     _       __     _                      _                          
                      /\ \ \___ _   _ _ __ __ _| |   /\ \ \___| |___      _____  _ __| | __                      
                     /  \/ / _ \ | | | '__/ _` | |  /  \/ / _ \ __\ \ /\ / / _ \| '__| |/ /                      
                    / /\  /  __/ |_| | | | (_| | | / /\  /  __/ |_ \ V  V / (_) | |  |   <                       
                    \_\ \/ \___|\__,_|_|  \__,_|_| \_\ \/ \___|\__| \_/\_/ \___/|_|  |_|\_\                      
                                                                                                                 

"""

    print(text)


# Call banner function to display title of program.
banner()

while True:
    print("1: Song classification instructions")
    print("2: Exit")

    choice = input("Please enter your choice: ")

    choice = choice.strip()

    if choice == "1":
        print("\nPlace your song into the 'input' folder. The format needs to be .wav.\n")
    elif choice == "2":
        break
    else:
        print("Error: Not a valid choice. Please retry.")
