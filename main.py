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
import time

# List of Tuning Parameters as Global Variables (To Do/Decide on)
# Global so that we keep them constant for all decision trees

#minSplit = 50
maxnumfeatures = 17 #calculated without dummies
forestsize = 300

class DecisionTree:

    def __init__(self, features, classification, splitmethod, maxdepth, minsampsplit, minimpuritydecrease, id):
        self.features = features
        #print(features.to_markdown())
        self.classification = classification

        #Tuning Parameters
        self.maxdepth = maxdepth
        self.minsampsplit = minsampsplit
        self.minimpuritydecrease = minimpuritydecrease

        self.classifier = tree.DecisionTreeClassifier(criterion = splitmethod, min_samples_split = self.minsampsplit, max_depth = self.maxdepth, min_impurity_decrease = self.minimpuritydecrease)
        self.numfeatures = np.random.randint(2, 17)
        self.featureslist = random.sample(range(17), self.numfeatures)
        #print(self.featureslist)
        self.usedfeatures = self.features.iloc[:, self.featureslist]
        #print(self.usedfeatures.to_markdown())
        self.id = id


    def buildTree(self):
        if (self.id % 20 == 0):
            print("Progress: built " + str(self.id) + " trees...")

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

    def __init__(self, numtrees, trainingfeat, trainingclass, method, maxdepth, minsampsplit, minimpuritydecrease):
        self.forest = []
        self.numtrees = numtrees
        self.trainingfeat = trainingfeat
        self.trainingclass = trainingclass
        self.splitmethod = method
        self.maxdepth = maxdepth
        self.minsampsplit = minsampsplit
        self.minimpuritydecrease = minimpuritydecrease

    def buildForest(self):
        self.starttime = time.time()

        print("Building Forest using the following methods:")
        print("Split method: " + str(self.splitmethod))
        print("Min Sample Split: " + str(self.minsampsplit))
        print("Max Depth: " + str(self.maxdepth))
        print("Min Impurity Decrease: " + str(self.minimpuritydecrease))

        for tree in range(self.numtrees):
            tempclassification = pd.DataFrame()
            tempfeatures = pd.DataFrame()

            for r in range(len(self.trainingclass)):
                index = np.random.randint(0, len(self.trainingclass))
                currfeatures = self.trainingfeat.iloc[[index]]
                currclass = self.trainingclass.iloc[[index]]
                tempclassification = pd.concat([tempclassification, currclass], axis = 0)
                tempfeatures = pd.concat([tempfeatures, currfeatures], sort=False, axis = 0)

            newtree = DecisionTree(tempfeatures, tempclassification, self.splitmethod, self.maxdepth, self.minsampsplit, self.minimpuritydecrease, tree)
            newtree.buildTree()
            #newtree.renderTree()
            self.forest.append(newtree)
        self.endtime = time.time()
        self.buildtime = self.endtime - self.starttime
        print("Completed Building Forest in " + str(self.buildtime) + " seconds\n")

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
        #self.predictionaccuracy = []
                                # Prediction, Guess
        self.numcorrect_pos = 0      #   1, 1
        self.numcorrect_neg = 0      #   0, 0
        self.numwrong_pos = 0        #   0, 1
        self.numwrong_neg = 0        #   1, 0

        #print(testingclass.to_markdown())

        for obsnum in range(len(testingclass)):
            observation = testingfeat.iloc[[obsnum]]
            obsclassactual = testingclass.iloc[[obsnum]].to_numpy()
            predictclass = self.classfiyObservation(observation, 1)
            #print("(" + str(predictclass) + ", " + str(obsclassactual) + ")")
            if predictclass == 1 and obsclassactual == 1:
                self.numcorrect_pos += 1
            elif predictclass == 0 and obsclassactual == 0:
                self.numcorrect_neg += 1
            elif predictclass == 1 and obsclassactual == 0:
                self.numwrong_neg += 1
            elif predictclass == 0 and obsclassactual == 1:
                self.numwrong_pos += 1

        self.accuracy = (self.numcorrect_neg + self.numcorrect_pos) / len(testingclass)

        self.printAccuracy()

    def printAccuracy(self):
        print("\n\nOVERALL FOREST STATISTICS\n=====================")
        print("Number of Trees: " + str(self.numtrees))
        print("Method Type: " + str(self.splitmethod))
        print("TOTAL ACCURACY: " + str(self.accuracy))
        print("\nDETAILED ACCURACY STATISTICS\n=====================")
        print("True Positives: " + str(self.numcorrect_pos))
        print("True Negatives: " + str(self.numcorrect_neg))
        print("False Positives: " + str(self.numwrong_neg))
        print("False Negatives: " + str(self.numwrong_pos))
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++\n")


def findOptimal(forest, trainingfeat, trainingclass, splitmethod, testingfeat, testingclass):
    #print("Hi")
    #Max Depth:
    maxdepthoptions = [2, 5, 8, 11, None]
    #TEST:
    #maxdepthoptions = [2, None]

    #Min Samples Split
    #minsamplessplitoptions = [2, 5]
    minsamplessplitoptions = [2, 5, 10, 20]

    #Min Impurity Decrease
    minimpuritydecreaserange = [0, 20, 2]


    #Default
    print("Building Default")
    currOptimalTree = RandomForest(args.forest, trainingfeat, trainingclass, method, None, 2, 0.0)
    currOptimalTree.buildForest()

    currOptimalTree.testAccuracy(testingfeat, testingclass)

    totaltreestesting = len(maxdepthoptions) * len(minsamplessplitoptions) * 10
    print("Total Number of Forests to Create: " + str(totaltreestesting))

    for maxdepth in maxdepthoptions:
        for minsampsplit in minsamplessplitoptions:
            for minimpuritydecrease in range(minimpuritydecreaserange[0], minimpuritydecreaserange[1], minimpuritydecreaserange[2]):
                minimp = minimpuritydecrease / 100
                rf = RandomForest(args.forest, trainingfeat, trainingclass, splitmethod, maxdepth, minsampsplit, minimp)
                rf.buildForest()
                rf.testAccuracy(testingfeat, testingclass)

                if rf.accuracy > currOptimalTree.accuracy:
                    currOptimalTree = rf

    print("\n\n\nOptimal Accuracy: " + str(currOptimalTree.accuracy))
    print("Optimal Methods: ")
    print("Max Depth: + " + str(currOptimalTree.maxdepth))
    print("Min Sample Split: " + str(currOptimalTree.minsampsplit))
    print("Minimum Impurity Decrease: " + str(currOptimalTree.minimpuritydecrease))
    print("Time to Build the Forest: " + str(currOptimalTree.buildtime))


if __name__ == '__main__':

        # trainclassification = pd.read_csv("trainingclass.csv")
        # trainfeatures = pd.read_csv("trainingfeat.csv")
        #
        # testclass = pd.read_csv("testingclass.csv")
        # testfeat = pd.read_csv("testingfeat.csv")
        parser = argparse.ArgumentParser()
        parser.add_argument("-f", "--forest", nargs='?', help="specify number of trees in forest (default = 300)", type=int, const=300)
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
            splitmethod = "entropy"
        elif args.entropy == 'gini':
            print("using gini")
            splitmethod = "gini"
        else:
            print("error: Method needs to be either gini or entropy")
            sys.exit()

        #findoptimalswitch: TRUE <- for finding the optimal values for tuning parameters
        #                   FALSE <- Application Use

        findoptimalswitch = False

        if findoptimalswitch == True:
            findOptimal(args.forest, trainingfeat, trainingclass, splitmethod, testingfeat, testingclass)

        else:
            #OPTIMAL VALUES FOUND FROM findOptimal():
            maxdepth = 8
            minsampsplit = 20
            minimpuritydecrease = 0.0

            rf = RandomForest(args.forest, trainingfeat, trainingclass, splitmethod, maxdepth, minsampsplit, minimpuritydecrease)
            rf.buildForest()
            rf.testAccuracy(testingfeat, testingclass)
