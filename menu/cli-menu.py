while True:

    print("Top-N Music Genre Classification Neural Network")
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