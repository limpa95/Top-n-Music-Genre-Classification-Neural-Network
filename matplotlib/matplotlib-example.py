import matplotlib.pyplot as plt


def read_output(name):
    with open(name, 'r') as file:
        string = file.read()
        string = string.strip()
        output_list = string.split(',')
        if name == "prediction.txt":
            output_list = [int(x) for x in output_list]
    return output_list


prediction_list = read_output("prediction.txt")
labels_list = read_output("labels.txt")
tuples_list = list(zip(labels_list, prediction_list))
sorted_tuples_list = sorted(tuples_list, key=lambda x: x[1])

y = [x[0] for x in sorted_tuples_list]
w = [x[1] for x in sorted_tuples_list]
c = ["blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue"]

plt.barh(y, w, 0.5, color=c)
plt.xlabel("Accuracy (%)")
plt.ylabel("Genre")
plt.title("Music Genre Classification")
plt.show()
