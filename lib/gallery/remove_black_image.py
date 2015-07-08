import sys
import cv2
import os


file_name = sys.argv[1]

file_new = list()
with open(file_name,'r') as f:
  for line in f:
    path = line.strip().split(' ')[0]
    if not os.path.exists(path):
      continue
    
    im = cv2.imread(path)
    if im.shape[0] == 100:
      continue
    
    file_new.append(line.strip())
    
with open("labels_noblack.txt",'w') as f:
  f.writelines("%s\n" % item for item in file_new )