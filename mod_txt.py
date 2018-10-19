import os
import re



# a = '245,2115,933,2115,933,2444,245,2444,0,"###" '
# print(a[-8])

# pattern = re.compile(r'"(.*)"')

names = os.listdir('./part2/')
print(names)
txt_names = [x for x in names if '.txt' in x ]
print(txt_names)
for txt in txt_names:
    txt_path = './part2/' + txt
    with open(txt_path, 'r', encoding= 'utf-8' ) as f:
        lines= f.readlines()
        # for line in lines:
            # sub_str = pattern.findall(line)
            # print(sub_str)
            # new_str = re.sub(pattern, "###", line)
            # print(new_str)
    with open(txt_path, 'w', encoding = 'utf-8') as f1:
        for line in lines:
            # sub_str = pattern.findall(line)
            # print(sub_str)
            new_str = line.replace("\"","").replace("\"","")

            # new_str = re.sub(pattern, r'###', line)
            # new_str = new_str[ : -10] + new_str[-8:]
            print(new_str)
            f1.write(new_str)


