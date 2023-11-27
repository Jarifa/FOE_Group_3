import pandas as pd
import os

def load_csv(folder_path):
    dataframes = []
    file = 1

    for filename in os.listdir(path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            dataframes.append(df)
            print("File",file,"complete")
            file += 1

    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

path = "Datasets"
df = load_csv(path)
print(path, "loaded into one df.")

pickle_file_path = os.path.join(path, 'combined_data.pk1')
df.to_pickle(pickle_file_path)
print(f"DataFrame saved at: {pickle_file_path}")