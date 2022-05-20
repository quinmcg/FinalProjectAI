from sklearn import tree
import graphviz
import numpy as np
import csv
import pandas as pd
import matplotlib
from tabulate import tabulate

# List of Tuning Parameters as Global Variables (To Do/Decide on)
# Global so that we keep them constant for all decision trees

minSplit = 50
numtrees = 1

class DecisionTree:

    def __init__(self, features, classification):
        self.features = features
        self.classification = classification
        self.classifier = tree.DecisionTreeClassifier(min_samples_split = minSplit)

    def buildTree(self):
        self.trainfeatures_dummy = pd.get_dummies(self.features)
        self.featuresarray = self.trainfeatures_dummy.to_numpy()
        self.classifarray = self.classification.to_numpy()
        self.classifier.fit(self.featuresarray, self.classifarray)

    def renderTree(self):
        tree.plot_tree(self.classifier)
        dot_data = tree.export_graphviz(self.classifier, out_file=None)
        graph = graphviz.Source(dot_data)
        graph.render("midd")




if __name__ == '__main__':

        trainclassification = pd.read_csv("relationshipstatus.csv")
        trainfeatures = pd.read_csv("trainingfeatures.csv")

        #print(trainfeatures.to_markdown())

        newtree = DecisionTree(trainfeatures, trainclassification)
        newtree.buildTree()
        newtree.renderTree()

        for i in range(numtrees):
            #sample our data
            tempclassification = pd.DataFrame()
            tempfeatures = pd.DataFrame()

            for r in range(len(trainclassification)):
                index = np.random.randint(0, len(trainclassification))
                currfeatures = trainfeatures.loc[index]
                currclass = trainclassification.loc[index]
                tempclassification = pd.concat([tempclassification, currclass])
                tempfeatures = pd.concat([tempfeatures, currfeatures])

            print(tempfeatures.to_markdown())

            newtree = DecisionTree(tempfeatures, tempclassification)
