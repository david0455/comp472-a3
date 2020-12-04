from MNBC import NB_Classifier
from NB_BOW_OV import getData

def main():
    nb = NB_Classifier()
    train_set = getData('./data/covid_training.tsv', False)
    test_set = getData('./data/covid_test_public.tsv', True)
    nb.fit(train_set)
    nb.predict(test_set)

if __name__ == '__main__':
    main()