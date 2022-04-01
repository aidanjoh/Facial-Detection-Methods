import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv (r'.\bounding_box_coords.csv', na_values = ['no info', '.'])
print (df.head)

print (df['Name'][0] +"string")

for i in range(len(df.index)):
  with open("Images/personal/"+ df["Name"][i], 'rb') as pgmf:
    im = plt.imread(pgmf)
    plt.imshow(im)
    plt.plot([df["x1"][i],df["x2"][i],df["x4"][i],df["x3"][i]],[df["y1"][i],df["y2"][i],df["y4"][i],df["y3"][i]], 'y')
    plt.show()
