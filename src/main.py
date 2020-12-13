from MNBC import NB_Classifier
from NB_BOW import getData


def main():
    nb_ov = NB_Classifier()
    nb_fv = NB_Classifier()

    train_set = getData('.//data/covid_training.tsv', False)
    test_set = getData('.//data/covid_test_public.tsv', True)

    nb_ov.fit_OV(train_set)
    nb_ov.predict(test_set)

    nb_fv.fit_FV(train_set)
    nb_fv.predict(test_set)


if __name__ == '__main__':
    main()
