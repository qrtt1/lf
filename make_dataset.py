import pandas as pd
from datasets import Dataset


def main():
    df = pd.read_excel('kautian.ods', engine='odf', sheet_name='例句')
    df = df.loc[:, ['漢字', '羅馬字', '華語']]

    dataset = Dataset.from_pandas(df)
    train_test_split = dataset.train_test_split(test_size=0.8)

    train_dataset = train_test_split['train']
    train_dataset.save_to_disk('dataset/train')

    test_dataset = train_test_split['test']
    test_dataset.save_to_disk('dataset/test')


if __name__ == '__main__':
    main()
