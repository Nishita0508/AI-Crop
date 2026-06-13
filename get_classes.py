import os

import os

class_names = sorted([
    d for d in os.listdir("dataset/Train")
    if os.path.isdir(os.path.join("dataset/Train", d))
])

print("Number of classes:", len(class_names))

print("Total classes:", len(class_names))

for i, c in enumerate(class_names):
    print(i, c)