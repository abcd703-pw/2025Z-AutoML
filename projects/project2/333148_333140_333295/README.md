# Mini_AutoML

This project was completed as part of the AutoML course at Warsaw University of Technology.

The goal of the project is to create a simplified AutoML system that allows automatic execution of a binary classification task on any provided dataset.


## File Structure:
```
├─ Datasets
│   ├─ for_portfolio
│   └─ for_tests
├─ src
│   ├─ preprocessing
│   ├─ temporary_code
│   │   ├─ info.md
│   │   └─ tmp_Karol.ipynb
│   └─ interfaces.py
├─ README.md
├─ manage.py
├─ .gitignore
└─ requirements.txt
```


# Plan działania
1) Parametry udostępnianego obiektu:
   - columns_to_ignore list[string]
   - date_use bin
   - time_resource
   - enseble_type
2) tabarena - parametry modeli
3) modele do portfolio
   - tabarena
   - woźnica
4) pipeline
  - preprocessing
      i) nie float + całkowicie różne wartości (próg ?, default =1) -> odrzucić
      ii) rozróżnienie zmiennych: bin, cat, num, date
          bin - 0,1
          cat - one hot (handle unknown = true)
          num - stnadard scaling - rozkład normalny(0,1)
          date - user określa użyć czy odrzucić (default odrzucić)
      iii) imutacja
          bin - unknown,  potencjalnie predykcja prostym modelem
          cat - unknown
          num - median, potencjalnie predykcja prostym modelem
          date - moda + warning
5) wybór modeli do stackingu - succesive halving (hyperband_pruner optuna) do top 5:
 - 1 etap: użycie hyperband z optuny osobno dla każdej rodziny modeli - u nas raczej tylko GBM i GLM (np z metryką weighted log-loss l_m) - wybór ok 15-20 modeli z przewagą GBM, np. 10 GBM, 5 GLM. \
   Uwaga implementacyjna: n_jobs=-1 należy ustawić tylko w optuna (chyba funkcja study.opitimize), wszędzie indziej podczas 1 etapu, w tym w modelach GBM, n_jobs=1
 - 2 etap: CV i wybór ok 10 najlepszych - koniecznie robust estimator - np median(l_m) + hiperparametr_kary * MAD(l_m) lub mean(l_m) + hiperparametr_kary * sd(l_m)
 - 3 etap: \
   i) Opcja 1: zachłanny wybór 5 modeli, np. poprzez maksymalizację wariancji rezyduów z regresji (liniowa z ridge to opcja), warunkowaną przez rezydua modeli juz dodanych do ensemble i na koniec hill-climbing \
   ii) Opcja 2: sprawdzenie wszystkich kombinacji (exhaustive search) - tylko jeśli będzie się wszystko wykonywało szybciej niż wymagane
6) ensemble - wybór rozwiązania z:
    -soft
    -hard
    -lin regression/ logistic regression
    -pseudo auto gluon 

Podział pracy:
ogólne zalecenia:
   - pakietowo, unikamy konfliktów

tabarena wydobycie - każdy poświęca min 1h, max 3h
preprocessing - Mikołaj
halving - Karol
ensemble - Ludwik

    














