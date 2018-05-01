import os
import numpy as np
import pandas as pd
import time 

from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.preprocessing import MultiLabelBinarizer

######################################### Reading and Splitting the Data ###############################################
main_dir = os.getcwd()
vect_dir = main_dir + '\\vectors'
write_dir = main_dir + '\\output_classifiers'

print os.listdir(vect_dir)
print os.listdir(write_dir)

big_s = 'mlp, rfc, best, best_score\n'

to_analyze = ['061.csv']

if len(to_analyze) == 0:
    to_analyze = os.listdir(vect_dir)

print to_analyze
for filename in to_analyze:
    if filename not in os.listdir(write_dir):

        # Read in all the data.
        data = pd.read_csv(vect_dir+'\\'+filename)

        s = 'mlp, rfc, best, best_score\n'
        classifiers = ['mlp', 'rfc']

        score=[]

        s_data = train_test_split(data, shuffle=True, random_state=100, test_size=0.30)

        # Separate out the x_data and y_data.
        x_train = s_data[0].loc[:, data.columns != "y"]
        y_train = s_data[0].loc[:, "y"]

        x_test = s_data[1].loc[:, data.columns != "y"]
        y_test = s_data[1].loc[:, "y"]
        save_test = y_test

        echantillons = 1000
        dimension = (len(y_train) + len(y_test))/echantillons
        print filename
        print dimension
        map_vect = {}



        def to_vect(pand):
            if pand in map_vect:
                return map_vect[pand]
            else:
                new_val = []
                for i in range(dimension):
                    if i == len(map_vect):
                        new_val.append(1)
                    else:
                        new_val.append(0)
                map_vect[pand] = new_val
                return map_vect[pand]

        ##def get_labels(old, new):
        ##    sol = {}
        ##    for i in old.index.values:
        ##        if len(sol) == dimension:
        ##            return map_vect
        ##        else:
        ##            print old[i]
        ##            print new[i]
        ##            if old[i] not in sol:
        ##                sol[old[i]] = new[i]
        ##    return sol
        for i in y_train.index.values:
            features = len(np.array(x_train.loc[[i]])[0,1:])
            break

        vect_nilf = [0]*features
        vect_nil = [0]*dimension

        new_xtrain = np.array([vect_nilf])
        new_ytrain = np.array([vect_nil])
        new_xtest = np.array([vect_nilf])
        new_ytest = np.array([vect_nil])

        for i in y_train.index.values:
            vect = to_vect(y_train[i])
            new_xtrain = np.vstack([new_xtrain, np.array(x_train.loc[[i]])[0,1:]])
            new_ytrain = np.vstack([new_ytrain, vect])

        for i in y_test.index.values:
            vect = to_vect(y_test[i])
            new_xtest = np.vstack([new_xtest, np.array(x_test.loc[[i]])[0,1:]])
            new_ytest = np.vstack([new_ytest, vect])

        ##y_train = MultiLabelBinarizer().fit_transform(y_train)
        ##y_test = MultiLabelBinarizer().fit_transform(y_test)

        x_train = new_xtrain
        y_train = new_ytrain

        x_test = new_xtest
        y_test = new_ytest

        #labels = get_labels(save_test, y_test)

        # ############################################### Multi Layer Perceptron #################################################
        print ('-- Multi Layer Perceptron --')
        mlp = MLPClassifier(hidden_layer_sizes=(178, 178, 178))
        mlp.fit(x_train, y_train)

        y_ptrain = mlp.predict(x_train)
        y_ptrain = y_ptrain.round()

        y_pred = mlp.predict(x_test)
        y_pred = y_pred.round()
        score.append(accuracy_score(y_test, y_pred))
        print ('Training Accuracy = '+str(accuracy_score(y_train, y_ptrain)))
        print ('Testing Accuracy = '+str(accuracy_score(y_test, y_pred))+'\n')

        # ############################################### Random Forest Classifier ##############################################
        print ('-- Random Forest Classifier --')
        rfc = RandomForestClassifier()
        rfc.fit(x_train, y_train)

        y_ptrain = rfc.predict(x_train)
        y_ptrain = y_ptrain.round()

        y_pred = rfc.predict(x_test)
        y_pred = y_pred.round()
        score.append(accuracy_score(y_test, y_pred))
        print ('Training Accuracy = '+str(accuracy_score(y_train, y_ptrain)))
        print ('Testing Accuracy = '+str(accuracy_score(y_test, y_pred))+' (No grid search)\n')

##        print ('Grid search :')
##        rfc = RandomForestClassifier()
##        parameters = {'n_estimators' : [1, 10, 20, 30], 'max_depth' : [2, 5, 10, 20]}
##        clf = GridSearchCV(rfc, parameters, cv = 10)
##        clf.fit(x_train, y_train)
##
##        score.append(accuracy_score(y_test, y_pred))

        maxi = np.argmax(score)
        score.append(classifiers[np.argmax(score)])
        score.append(score[maxi])

        for val in score:
            big_s +=str(val)
            big_s +=','
            s+=str(val)
            s+= ','
        s = s[:-1]
        big_s = big_s[:-1]
        s+= '\n'
        big_s+= '\n'

        fich = open(write_dir+'\\'+filename, 'w')
        fich.write(s)
        fich.close()
        
fich = open(write_dir+'\\all.csv', 'w')
fich.write(big_s)
fich.close()



