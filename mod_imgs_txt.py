
import os

names = os.listdir('./part2/')
print(names)
txt_names = [x for x in names if '.txt' in x ]
print(txt_names)
for txt in txt_names:
    txt_path = './part2/' + txt
    with open(txt_path, 'r', encoding= 'utf-8' ) as f:
        lines = f.readlines()
    with open(txt_path, 'w') as f1:
        for line in lines:
            line = line.split(',')
            line = line[:8] + [line[-1]]
            new_line = ','.join(line)
            print(new_line)
            f1.write(new_line)


