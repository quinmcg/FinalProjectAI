from sklearn import tree
import graphviz
import numpy as np
import csv
import pandas as pd
import matplotlib

class Main:



    def decisionTree():
        # with open("relationshipstatus.csv", mode = 'r') as file_name:
        #     trainclassification = csv.reader(file_name)
        #
        # with open("trainingfeatures.csv", mode = 'r') as file_name:
        #     trainfeatures = csv.reader(file_name)
        # for lines in trainfeatures:
        #     print(lines)

        trainclassification = pd.read_csv("relationshipstatus.csv")
        trainfeatures = pd.read_csv("trainingfeatures.csv")
        #find_partner	gender	gpa	class	siblings	parents_married	race	housing	year
        trainfeatures_dummy = pd.get_dummies(trainfeatures, columns = ['find_partner', 'gender', 'class', 'parents_married', 'race', 'housing', 'year'])
        featuresarray = trainfeatures_dummy.to_numpy()
        classifarray = trainclassification.to_numpy()
        #print(trainfeatures_dummy.loc[0])

        classifier = tree.DecisionTreeClassifier(min_samples_split = 50)
        classifier.fit(featuresarray, classifarray)

        tree.plot_tree(classifier)
        dot_data = tree.export_graphviz(classifier, out_file=None)
        graph = graphviz.Source(dot_data)
        graph.render("midd")

if __name__ == '__main__':


    Main.decisionTree()
