import matplotlib.pyplot as plt

y = ["Hip-hop", "Blues", "Classical", "Jazz", "Disco", "Reggae", "Country", "Pop", "Metal", "Rock"]
w = [0, 0, 5, 10, 10, 20, 20, 50, 60, 80]
c = ["blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue",]

plt.barh(y, w, 0.5, color = c)
plt.xlabel("Accuracy (%)")
plt.ylabel("Genre")
plt.title("Music Genre Classification")
plt.show()