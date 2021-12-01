import pandas as pd
import matplotlib as mat

heartData = pd.read_csv('heart.csv')
heartData.hist(figsize=(12,12))
