import pandas as pd
import cv2
import pathlib
from collections import namedtuple
import pickle

data = pd.read_csv('D:/urgence_sante/my_copy.csv', index_col = 'NoRIP', keep_default_na = False)

data_root = pathlib.Path("C:/urgences_sante_run_sheets/batch-crop/batch")
paths = list(data_root.glob('*'))
names = [x.stem.split("_")[0:2] for x in paths]
ids, hour_nums = zip(*names)
ids = list(map(int, ids))
columns = ['sHrSV' + hour_num.strip('hour') for hour_num in hour_nums]

times = data.lookup(ids, columns)
hours = [time.split(":")[0] for time in times]
images = [cv2.imread(str(path), cv2.COLOR_BGR2GRAY) for path in paths]

TimeImage = namedtuple("TimeImage", ["hour", "image", "path"])
hours_images_paths = [h for h in  map(TimeImage, hours, images, paths) if h.hour != ""]

with open('hours.pkl', 'wb') as file:
  pickle.dump(hours_images_paths, file)

with open('hours.pkl', 'rb') as file:
  b = pickle.load(file)

file = open("hour_pickle.pkl",'wb') 
pickle.dump(hours_images_paths, file)
file.close()

file = open("hour_pickle.pkl", 'r')
pickle.load(file)