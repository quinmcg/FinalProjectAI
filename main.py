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

    def classifyTree(self, observation):
        usedobs = observation.iloc[:, self.featureslist]
        usedobsnp = usedobs.to_numpy()
        print("USED OBS NP:")
        print(usedobsnp)
        prediction = self.classifier.predict(usedobsnp)
        print(prediction)
        return prediction

class RandomForest:

    def __init__(self, numtrees, trainingfeat, trainingclass):
        self.forest = []
        self.numtrees = numtrees
        self.trainingfeat = trainingfeat
        self.trainingclass = trainingclass

    def buildForest(self):
        print("Build")


        for tree in range(self.numtrees):
            tempclassification = pd.DataFrame()
            tempfeatures = pd.DataFrame(columns = collist)

            for r in range(len(self.trainingclass)):
            #for r in range(10):
                index = np.random.randint(0, len(self.trainingclass))
                currfeatures = self.trainingfeat.iloc[[index]]
                currclass = self.trainingclass.iloc[[index]]
                tempclassification = pd.concat([tempclassification, currclass], axis = 0)
                tempfeatures = pd.concat([tempfeatures, currfeatures], sort=False, axis = 0)

            #print(tempfeatures.to_markdown())
            #print(tempclassification.to_markdown())
            newtree = DecisionTree(tempfeatures, tempclassification, tree)
            newtree.buildTree()
            newtree.renderTree()
            self.forest.append(newtree)
        print(len(self.forest))

    def classfiyObservation(self, observation):
        print("Classify:")
        vote1 = 0
        vote0 = 0
        for tree in self.forest:
            prediction = tree.classifyTree(observation)
            if prediction == 1:
                print("1")
            else:
                print("O")


if __name__ == '__main__':

        # trainclassification = pd.read_csv("trainingclass.csv")
        # trainfeatures = pd.read_csv("trainingfeat.csv")
        #
        # testclass = pd.read_csv("testingclass.csv")
        # testfeat = pd.read_csv("testingfeat.csv")

        features = pd.read_csv("trainingfeatures.csv")
        classification = pd.read_csv("relationshipstatus.csv")

        featuresdummy = pd.get_dummies(features)

        testingfeat = featuresdummy.iloc[850:]
        trainingfeat = featuresdummy.iloc[:850]

        trainingclass = classification.iloc[:850]
        testingclass = classification.iloc[850:]

        #print(trainingfeat.to_markdown())

        #newtree = DecisionTree(trainfeatures, trainclassification)
        #newtree.buildTree()
        #newtree.renderTree()

        rf = RandomForest(2, trainingfeat, trainingclass)
        rf.buildForest()

        obs = testingfeat.iloc[[5]]
        print(obs)
        obsclass = testingclass.iloc[[5]].to_numpy()

        print(str(rf.classfiyObservation(obs)))
        print(str(obsclass))
