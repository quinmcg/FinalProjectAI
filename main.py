from sklearn import tree
import graphviz
import numpy as np
import csv
import pandas as pd
import matplotlib
from tabulate import tabulate
import random
import sys, getopt
import argparse

# List of Tuning Parameters as Global Variables (To Do/Decide on)
# Global so that we keep them constant for all decision trees

minSplit = 50
maxnumfeatures = 17 #calculated without dummies
forestsize = 300

class DecisionTree:

    def __init__(self, features, classification, method, id):
        self.features = features
        #print(features.to_markdown())
        self.classification = classification
        self.classifier = tree.DecisionTreeClassifier(criterion = method, min_samples_split = minSplit)
        self.numfeatures = np.random.randint(2, 17)
        self.featureslist = random.sample(range(17), self.numfeatures)
        #print(self.featureslist)
        self.usedfeatures = self.features.iloc[:, self.featureslist]
        #print(self.usedfeatures.to_markdown())
        self.id = id

    def buildTree(self):
        if (self.id % 10 == 0):
            print("Building Tree #" + str(self.id))
     
        self.featuresarray = self.usedfeatures.to_numpy()
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
        prediction = self.classifier.predict(usedobsnp)
        return prediction

class RandomForest:

    def __init__(self, numtrees, trainingfeat, trainingclass, method):
        self.forest = []
        self.numtrees = numtrees
        self.trainingfeat = trainingfeat
        self.trainingclass = trainingclass
        self.method = method

    def buildForest(self):
        print("Building Forest using " + self.method + " method")

        for tree in range(self.numtrees):
            tempclassification = pd.DataFrame()
            tempfeatures = pd.DataFrame()

            for r in range(len(self.trainingclass)):
                index = np.random.randint(0, len(self.trainingclass))
                currfeatures = self.trainingfeat.iloc[[index]]
                currclass = self.trainingclass.iloc[[index]]
                tempclassification = pd.concat([tempclassification, currclass], axis = 0)
                tempfeatures = pd.concat([tempfeatures, currfeatures], sort=False, axis = 0)

            newtree = DecisionTree(tempfeatures, tempclassification, self.method, tree)
            newtree.buildTree()
            #newtree.renderTree()
            self.forest.append(newtree)
        #print(len(self.forest))

    def classfiyObservation(self, observation, method):
        #print("Classify:")
        vote1 = 0
        vote0 = 0
        for tree in self.forest:
            #print("ID: " + str(tree.id))
            #print("NumFeatures: " + str(tree.numfeatures))
            prediction = tree.classifyTree(observation)
            if prediction == 1:
                #print("1")
                vote1+=1
            else:
                #print("O")
                vote0+=1
        #method: indicates whether we want it to return average (ie probability it is 1), or max
        #MAX:
        if method == 1:
            if vote0 >= vote1:
                return 0
            else:
                return 1
        #AVERAGE
        else:
            return round((vote0+vote1)/self.numtrees)

    def testAccuracy(self, testingfeat, testingclass):
        self.predictionaccuracy = []
                                # Prediction, Guess
        numcorrect_pos = 0      #   1, 1
        numcorrect_neg = 0      #   0, 0
        numwrong_pos = 0        #   0, 1
        numwrong_neg = 0        #   1, 0

        #print(testingclass.to_markdown())

        for obsnum in range(len(testingclass)):
            observation = testingfeat.iloc[[obsnum]]
            obsclassactual = testingclass.iloc[[obsnum]].to_numpy()
            predictclass = self.classfiyObservation(observation, 1)
            #print("(" + str(predictclass) + ", " + str(obsclassactual) + ")")
            if predictclass == 1 and obsclassactual == 1:
                numcorrect_pos += 1
            elif predictclass == 0 and obsclassactual == 0:
                numcorrect_neg += 1
            elif predictclass == 1 and obsclassactual == 0:
                numwrong_neg += 1
            elif predictclass == 0 and obsclassactual == 1:
                numwrong_pos += 1
        accuracy = (numcorrect_neg + numcorrect_pos) / len(testingclass)
        print("\n\nOVERALL FOREST STATISTICS\n=====================")
        print("Number of Trees: " + str(self.numtrees))
        print("Method Type: " + str(self.method))
        print("TOTAL ACCURACY: " + str(accuracy))
        print("\nDETAILED ACCURACY STATISTICS\n=====================")
        print("True Positives: " + str(numcorrect_pos))
        print("True Negatives: " + str(numcorrect_neg))
        print("False Positives: " + str(numwrong_neg))
        print("False Negatives: " + str(numwrong_pos))
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++\n")




if __name__ == '__main__':

        # trainclassification = pd.read_csv("trainingclass.csv")
        # trainfeatures = pd.read_csv("trainingfeat.csv")
        #
        # testclass = pd.read_csv("testingclass.csv")
        # testfeat = pd.read_csv("testingfeat.csv")
        parser = argparse.ArgumentParser()
        parser.add_argument("-f", "--forest", nargs='?', help="specify forrest size (default = 300)", type=int, const=300)
        parser.add_argument("-u", "--user", help = "-u for user input test case", action="store_true")
        parser.add_argument("-m", "--entropy", nargs='?', help = "-m to specify method, so far entropy and gini", type=str, const="gini")
        args = parser.parse_args()

        features = pd.read_csv("trainingfeatures.csv")
        classification = pd.read_csv("relationshipstatus.csv")

        featuresdummy = pd.get_dummies(features)
        trainingfeat = featuresdummy.iloc[:900]
        trainingclass = classification.iloc[:900]


        if args.user:
            #user goes through survey
            print("args.user")
        else:
            testingfeat = featuresdummy.iloc[900:]
            testingclass = classification.iloc[900:]

        if args.entropy == 'entropy':
            print("using entropy")
            method = "entropy"
        elif args.entropy == 'gini':
            print("using gini")
            method = "gini"
        else:
            print("error: Method needs to be either gini or entropy")
            sys.exit()

        rf = RandomForest(args.forest, trainingfeat, trainingclass, method)
        rf.buildForest()


        rf.testAccuracy(testingfeat, testingclass)

