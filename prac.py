import pandas as pd
from glob import glob
import torch

if __name__ == '__main__':
    '''
    data_dir = '/media/data1/jeon/workspace/'
    data_file = '/media/data1/jeon/workspace/patient_list.xlsx'
    df = pd.read_excel(data_file, sheet_name='train_val_test_idx')
    print(len(df))
    #print(df.index)
    #print(df.values)
    filepath_idx = df.index[df[0] == 0]

    print(filepath_idx)

    df2 = pd.read_excel(data_file, sheet_name='filenames')
    df2 = pd.DataFrame(df2)
    filepath = df2.loc[filepath_idx, :].values.tolist()
    #filepath = df2.loc[filepath_idx, :]
    #print(filepath.columns[0])
    print(filepath[0][0])
    print(filepath[0][0].split('/'))
    img_path = filepath[0][0].replace("C:/Workspace/", data_dir)
    mask = img_path.split('/')[0:-1]
    mask_path = '/'.join(mask) + '/Col*'
    print(mask_path)
    mask_file = glob(mask_path)
    print(mask_file)
    '''
    pred = torch.rand(1, 5, 20, 224, 224)
    truth = torch.rand(1, 5, 20, 224, 224)
    # print(pred.shape)

    pred = pred.squeeze(0)  # [5, 20, 224, 224]
    truth = truth.squeeze(0)
