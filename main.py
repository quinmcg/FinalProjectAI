from sklearn import tree
import graphviz
import numpy as np
import csv
import pandas as pd
import matplotlib
from tabulate import tabulate
import random

# List of Tuning Parameters as Global Variables (To Do/Decide on)
# Global so that we keep them constant for all decision trees

minSplit = 50
maxnumfeatures = 17 #calculated without dummies

collist = ["midd_find_relationship", "midd_find_hookup", "midd_goes_relationship" ,"midd_goes_hookup", "midd_lookingfor_relationship", "midd_lookingfor_hookup",	"midd_opps_newpeople", "mrtl_potential_date", "find_partner", "gender", "gpa", "class", "siblings", "parents_married", "race", "housing", "year"]


class DecisionTree:

    def __init__(self, features, classification, id):
        self.features = features
        self.classification = classification
        self.classifier = tree.DecisionTreeClassifier(min_samples_split = minSplit)
        self.numfeatures = np.random.randint(2, 17)
        self.featureslist = random.sample(range(17), self.numfeatures)
        #print(self.featureslist)
        self.usedfeatures = self.features.iloc[:, self.featureslist]
        #print(self.usedfeatures.to_markdown())
        self.id = id

    def buildTree(self):

        self.trainfeatures_dummy = pd.get_dummies(self.usedfeatures)
        self.featuresarray = self.trainfeatures_dummy.to_numpy()
        self.classifarray = self.classification.to_numpy()
        self.classifier.fit(self.featuresarray, self.classifarray)

    def renderTree(self):
        tree.plot_tree(self.classifier)
        dot_data = tree.export_graphviz(self.classifier, out_file=None)
        graph = graphviz.Source(dot_data)
        graph.render(str(self.id))

class RandomForest:

    def __init__(self, numtrees):
        self.forest = []
        self.numtrees = numtrees


    def buildForest(self):
        print("Build")

        for tree in range(self.numtrees):
            tempclassification = pd.DataFrame()
            tempfeatures = pd.DataFrame(columns = collist)

            for r in range(len(trainclassification)):
            #for r in range(10):
                index = np.random.randint(0, len(trainclassification))
                currfeatures = trainfeatures.iloc[[index]]
                currclass = trainclassification.iloc[[index]]
                tempclassification = pd.concat([tempclassification, currclass], axis = 0)
                tempfeatures = pd.concat([tempfeatures, currfeatures], sort=False, axis = 0)

            #print(tempfeatures.to_markdown())
            #print(tempclassification.to_markdown())
            newtree = DecisionTree(tempfeatures, tempclassification, tree)
            newtree.buildTree()
            newtree.renderTree()
            self.forest.append(newtree)
        print(len(self.forest))

    def classfiyObservation(self):
        print("Classify:")


if __name__ == '__main__':

        trainclassification = pd.read_csv("trainingclass.csv")
        trainfeatures = pd.read_csv("trainingfeat.csv")

        testclass = pd.read_csv("testingclass.csv")
        testfeat = pd.read_csv("testingfeat.csv")



        #print(trainfeatures.to_markdown())

        #newtree = DecisionTree(trainfeatures, trainclassification)
        #newtree.buildTree()
        #newtree.renderTree()

        rf = RandomForest(2)
        rf.buildForest()
