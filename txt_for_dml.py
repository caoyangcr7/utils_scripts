import os

path = './test_data_for_dml'
image_classes = os.listdir(path)
print(image_classes)

with open('train.txt', 'w', encoding='utf8') as f:
    for image_class in image_classes:
        new_path = os.path.join(path, image_class)
        for img in os.listdir(new_path):
            f.write(str(img) + ' ' + str(image_class))
            f.write('\n')




