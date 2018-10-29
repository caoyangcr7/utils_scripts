import scipy.io as sio

data = sio.loadmat("./test_data/cars_meta.mat")
print(data["__version__"])
print(data.keys())

print(data["class_names"][0][0])
print(data["class_names"][0])
print(len(data["class_names"][0]))
# print(data)
# print(type(data["annotations"][0][0]))