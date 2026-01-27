import numpy as np
import pandas as pd
import time
import json
import catboost
import random
import sklearn
from sklearn.ensemble import  VotingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn import set_config
from skopt.space import Integer, Categorical, Real
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer, make_column_selector, ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

class KGBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, budget=1200, scorer="balanced_accuracy"):
        self.budget = budget
        self.scorer = scorer
        super().__init__()

    def fit(self, X, y):
        y = np.ravel(y)
        ### INICJALIZACJA FITA
        ###### WCIAGAMY ZBIORY, ŁADUJEMY MODELE, BIERZEMY ZMIENNE JAKOŚCIOWE DO CATBOOSTA
        total_start = time.time() # Na przyszłość, by sprawdzić, ile mamy czasu
        with open("models.json", "r") as configs: # Ładujemy wszystko
            models = json.load(configs)
        #print(type(X))
        categories = X.select_dtypes(include=["object", "category"]).columns
        cat_indices = X.columns.get_indexer(categories)
        results_cat = [] # Tu wrzucimy wyniki catboostów w celu wyboru najlepszego modelu
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

        # Sprawdzenie rozmiaru dancyh
        if X.shape[0]<=700:
            size=0
        elif X.shape[0]<=5000:
            size=1
        else:
            size=2
        if X.shape[1]>50:
            #size=3
            pass

        # PREPROCESSING

        # Dla niecatboostów

        ## Obsługa zmiennych numerycznych
        num_pipeline = Pipeline(steps=[
            ('impute', SimpleImputer()),
            ('scale', MinMaxScaler())
        ])

        ## Obsługa zmiennych kategorycznych
        cat_pipeline = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('one_hot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
        ])

        col_trans = ColumnTransformer(
            transformers=[
                (
                    "numeric_preprocessing",
                    num_pipeline,
                    make_column_selector(dtype_include=np.number),
                ),
                (
                    "categorical_preprocessing",
                    cat_pipeline,
                    make_column_selector(dtype_include=["object", "category", "bool"]),
                ),
            ],
            remainder="passthrough",
        )

        ## Pipeline

        model_pipeline = Pipeline(steps=[
            ('preprocessing', col_trans)
        ])



        # Dla catboostów

        ## Obsługa zmiennych numerycznych
        num_pipeline_3 = Pipeline(steps=[
            ('impute', SimpleImputer(strategy="most_frequent"))
        ])

        ## Obsługa zmiennych kategorycznych
        cat_pipeline_3 = Pipeline(steps=[
            ('impute',SimpleImputer(strategy = "most_frequent"))
        ])

        col_trans_3 = ColumnTransformer(
            transformers=[
                (
                    "numeric_preprocessing",
                    num_pipeline_3,
                    make_column_selector(dtype_include=np.number),
                ),
                (
                    "categorical_preprocessing",
                    cat_pipeline_3,
                    make_column_selector(dtype_include=["object", "category", "bool"]),
                ),
            ],
            remainder="passthrough",
        )

        ## Pipeline

        model_pipeline_3 = Pipeline(steps=[
            ('preprocessing', col_trans_3)
        ])



        le = LabelEncoder()

        y_train_p = le.fit_transform(y_train)
        X_train_p = model_pipeline.fit_transform(X_train, y_train_p)
        X_test_p = model_pipeline.transform(X_test)
        y_test_p = le.fit_transform(y_test)

        y_train_p_cat = le.fit_transform(y_train)
        X_train_p_cat = model_pipeline_3.fit_transform(X_train, y_train_p_cat)
        X_test_p_cat = model_pipeline_3.transform(X_test)
        y_test_p_cat = le.fit_transform(y_test)


        ### POJEDYNCZE WYKONANIE WSZYSTKIEGO POZA PĘTLĄ - W CELU USTALENIA ILE MOŻEMY ZAPUŚCIĆ MODELI

        #start = time.time() 
        #cls = eval(models[0]["class"]) # robimy zaciąg z napisu
        #obj = Pipeline([
        #    ("impute", SimpleImputer(strategy="most_frequent")),
        #    ("catpred", cls(cat_features = cat_indices, **models[0]["params"]))]) # używamy zaciągu jako modelu (catboost)
        #obj.fit(X_train, y_train)
        #sc = balanced_accuracy_score(y_test, obj.predict(X_test)) # wynik na kroswalidacji
        #print(f"model 0: {sc}")
        #results_cat.append(sc) # dodawańsko
        #end = time.time()
        ### UŻYWAMY METODY ROZSĄDNEJ, ABY WYBRAĆ, NA ILE MODELI STARCZY NAM CZASU

        #benchmark = end-start
        #print(f"time elapsed: {benchmark}")
        #approx_list = 15*[benchmark] + 15*[2*benchmark] + 15*[20*benchmark] + [100*benchmark] # więcej estymatorów -- więcej czasu
        #upper_bound = sum(np.cumsum(approx_list)<(self.budget*3/4))-1 # działa. NIE TYKAJCIE TEGO, pozostałe modele będą się liczyły w tzw marginesie
        # margines - 1/4 budżetu (ustalone metodą bo tak)
        #print(f"Setting upper bound for {upper_bound} with 1/4 time in reserve")

        ### ROBIMY WSZYSTKO DWOMA WHILE

        i = 0
        # Tworzymy catboosty, dopóki nie zostanie mniej niż 6 minut
        while (self.budget - time.time() + total_start)>(self.budget/4+60) and (i<38):
            start = time.time() 
            cls = eval(models[i]["class"])
            obj=cls(cat_features=cat_indices, **models[i]["params"])
            obj.fit(X_train_p_cat, y_train_p_cat)
            sc = balanced_accuracy_score(y_test_p_cat, obj.predict(X_test_p_cat))
            results_cat.append(sc)
            end = time.time() 
            print(f"model {i}: {sc:.4f}, Elapsed time: {(end - start):.4f} seconds")
            print(f"Time left: {(self.budget - time.time() + total_start):.4f} seconds")

            i+=1
        cat=i # liczba catboostów
        print(f"Fitted {cat} catboost models")
        print("Proceeding to other models")
        # Pozsotałe modele
        results=[0]*100
        j = size
        # Tworzymy inne modele, dopóki nie zostanie mniej niż 4 minuty
        # Tworzymy modele zgodnie z wielkością danych (np. jeśli dane są średnie to najpierw policzą się modele dla średnich danych),
        # a jeśli starczy czasu to inne
        i = 38
        while (self.budget - time.time() + total_start)>(self.budget/5) and (i<50):

            start = time.time()
            cls = eval(models[38+j]["class"])
            obj = cls(**(models[38+j]["params"]))
            obj.fit(X_train_p, y_train_p)
            sc = balanced_accuracy_score(y_test_p, obj.predict(X_test_p))
            results[38+j]=sc
            end = time.time()
            print(f"model {i}: {sc:.4f}, Elapsed time: {(end - start):.4f} seconds")
            print(f"Time left: {(self.budget - time.time() + total_start):.4f} seconds")

            i += 1

            start = time.time()
            cls = eval(models[38 + j + 4]["class"])
            obj = cls(**(models[38 + j + 4]["params"]))
            obj.fit(X_train_p, y_train_p)
            sc = balanced_accuracy_score(y_test_p, obj.predict(X_test_p))
            results[38+j+4]=sc
            end = time.time()
            print(f"model {i}: {sc:.4f}, Elapsed time: {(end - start):.4f} seconds")
            print(f"Time left: {(self.budget - time.time() + total_start):.4f} seconds")

            i += 1

            start = time.time()
            cls = eval(models[38 + j + 8]["class"])
            obj = cls(**(models[38 + j + 8]["params"]))
            obj.fit(X_train_p, y_train_p)
            sc = balanced_accuracy_score(y_test_p, obj.predict(X_test_p))
            results[38+j+8]=sc
            end = time.time()
            print(f"model {i}: {sc:.4f}, Elapsed time: {(end - start):.4f} seconds")
            print(f"Time left: {(self.budget - time.time() + total_start):.4f} seconds")

            i += 1
            j=(j+1)%4
        print(f"Fitted {i-37} other models")
        print("Building Ensembles")
        ### pętla for na benchmarku, spróbujemy whilem to zrobić
        #for i in range(1, upper_bound): 
        #    cls = eval(models[i]["class"])
        #    obj = Pipeline([
        #    ("impute", SimpleImputer(strategy="most_frequent")),
        #    ("catpred", cls(cat_features = cat_indices, **models[i]["params"]))])
        #    obj.fit(X_train, y_train)
        #    sc = balanced_accuracy_score(y_test, obj.predict(X_test))
        #    results_cat.append(sc)
        #    print(f"model {i}: {sc}")


        best_model_index_cat = results_cat.index(max(results_cat)) # wybieramy naszego szefa
        best_model_index_other = results.index(max(results))  # najlepszy wśród niecatboostów
        best_estimator = eval(models[best_model_index_cat]["class"])(cat_features = cat_indices, **models[best_model_index_cat]["params"]) # budujemy szefa
        best_estimator_other = eval(models[best_model_index_other]["class"])( **models[best_model_index_other]["params"])


        # Tworzymy listę do stackingu i votingu. Trzy losowe catboosty + najlepszy niecatboost + losowy niecatboost (do stackingu)
        # lub najlepszy catboost + dwa losowe catboosty + najlepszy niecatboost + losowy niecatboost (do votingu)
        catlist = list(range(0, cat))
        if best_model_index_cat in catlist:
            catlist.remove(best_model_index_cat)

        estimators_list_catboost = [(models[i]["name"]+"_"+str(i),eval(models[i]["class"])(cat_features = None, **models[i]["params"])) for i in random.sample(catlist, 3)]

        noncat = list(range(38, 50))
        if best_model_index_other in noncat:
            noncat.remove(best_model_index_other)

        ran=random.sample(noncat,1)[0]
        estimators_list=estimators_list_catboost+\
        [(models[best_model_index_other]["name"]+"_"+str(best_model_index_other),eval(models[best_model_index_other]["class"])(**models[best_model_index_other]["params"]))]+ \
                        [(models[ran]["name"] + "_" + str(ran),
                          eval(models[ran]["class"])(**models[ran]["params"]))]

        estimators_list_voting=estimators_list
        best_cat_ensemble = eval(models[best_model_index_cat]["class"])(**models[best_model_index_cat]["params"])
        estimators_list_voting[0]=(models[best_model_index_cat]["name"]+"_"+str(best_model_index_cat),best_cat_ensemble)


        # Stacking, voting (hard), voting (soft)
        sc_stack_boost=0
        sc_vote_boost=0
        sc_vote_boost_soft=0
        if (self.budget - time.time() + total_start)>(self.budget/10):
            print("Building Stacking Classifier")
            start = time.time()
            stack_boost =  StackingClassifier(estimators = estimators_list,
                                           final_estimator=eval(models[best_model_index_cat]["class"])(**models[best_model_index_cat]["params"])) #stacking jeden jest zrobiony
            stack_boost.fit(X_train_p, y_train_p)
            sc_stack_boost = balanced_accuracy_score(y_test_p, stack_boost.predict(X_test_p))
            end = time.time()
            rough_benchmark = end - start
            print(f"Stacking: {sc_stack_boost: .4f}, Elapsed time: {(end - start):.4f} seconds")
            print(f"Time left: {(self.budget - time.time() + total_start):.4f} seconds")


            if (self.budget - time.time() + total_start)>rough_benchmark:
                print("Building Hard Voting")
                start = time.time()
                vote_boost = VotingClassifier(estimators=estimators_list_voting)
                vote_boost.fit(X_train_p, y_train_p)
                sc_vote_boost = balanced_accuracy_score(y_test_p, vote_boost.predict(X_test_p))
                end = time.time()
                print(f"Voting (hard): {sc_vote_boost: .4f}, Elapsed time: {(end - start):.4f} seconds")
                print(f"Time left: {(self.budget - time.time() + total_start):.4f} seconds")


                if (self.budget - time.time() + total_start)>rough_benchmark:
                    print("Building Soft Voting")
                    start = time.time()
                    vote_boost_soft = VotingClassifier(estimators=estimators_list_voting, voting="soft")
                    vote_boost_soft.fit(X_train_p, y_train_p)
                    sc_vote_boost_soft = balanced_accuracy_score(y_test_p,
                                                                vote_boost_soft.predict(X_test_p))
                    end = time.time()
                    print(f"Voting (soft): {sc_vote_boost_soft: .4f}, Elapsed time: {(end - start):.4f} seconds")
                    print(f"Time left: {(self.budget - time.time() + total_start):.4f} seconds")

        # Preprocessing dla całego zbioru danych
        print("Finished fitting, deciding on model")


        # Sprawdzamy, który model lub ensemble okazał się najlepszy
        max_index=[max(results_cat),max(results),sc_stack_boost,sc_vote_boost,sc_vote_boost_soft].index(max([max(results_cat),max(results),sc_stack_boost,sc_vote_boost,sc_vote_boost_soft]))

        if max_index==0:
            self.best_model = Pipeline([("preprocessing",model_pipeline_3),
                                        ("estimator", best_estimator)])
            self.best_model.fit(X, y)
            print(f"The best model is model number {best_model_index_cat}.")
            print(models[best_model_index_cat]["name"])
        elif max_index == 1:
            self.best_model = Pipeline([("preprocessing",model_pipeline),
                                        ("estimator", best_estimator_other)])
            self.best_model.fit(X, y)
            print(f"The best model is model number {best_model_index_other}.")
            print(models[best_model_index_other]["name"])
        elif max_index == 2:
            #stack_boost.fit(X_p, y_p)
            print(f"The best model is stacking.")
            self.best_model = Pipeline([("preprocessing",model_pipeline),
                                        ("estimator", stack_boost)])
            self.best_model.fit(X, y)
        elif max_index==3:
            #vote_boost.fit(X_p, y_p)
            print(f"The best model is voting (hard).")
            #self.best_model = vote_boost
            self.best_model = Pipeline([("preprocessing",model_pipeline),
                                        ("estimator", vote_boost)])
            self.best_model.fit(X, y)
        else:
            #vote_boost_soft.fit(X_p, y_p)
            print(f"The best model is voting (soft).")
            self.best_model = Pipeline([("preprocessing",model_pipeline),
                                        ("estimator", vote_boost_soft)])
            self.best_model.fit(X, y)
        self.is_fitted_ = True

        return self
    


    ### Dwie proste funkcje na predict i predict_proba
    def predict(self, X):
        return self.best_model.predict(X)
    
    def predict_proba(self, X):
        return self.best_model.predict_proba(X)


        



        
        
