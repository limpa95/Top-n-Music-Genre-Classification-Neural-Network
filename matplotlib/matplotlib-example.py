import matplotlib.pyplot as plt
import json


def read_output(name):
    with open(name, 'r') as file:
        json_dict = json.load(file)
    return json_dict


output_dict = read_output("output.json")
tuples_list = list(output_dict.items())
sorted_tuples_list = sorted(tuples_list, key=lambda x: x[1])

y = [x[0] for x in sorted_tuples_list]
w = [x[1] for x in sorted_tuples_list]
c = ["blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue"]

plt.barh(y, w, 0.5, color=c)
plt.xlabel("Accuracy (%)")
plt.ylabel("Genre")
plt.title("Music Genre Classification")
plt.show()
