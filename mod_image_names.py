import os

path = './part2/'
names = os.listdir(path)
for name in names:
    old_name = path + name
    new_name =  path + name.replace(name.split('_')[0], 'img')
    os.rename(old_name, new_name)
