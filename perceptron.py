import argparse
import sys
import getopt
import math
import re
import random
from random import randint

#Main----------------------------------------------------------
def main(argv):
	trainingDatasets = ["speeches.train.liblinear"]
	devDatasets = ["speeches.train.liblinear"]
	testDatasets = ["speeches.test.liblinear"]

	train, num = parseFile(trainingDatasets)
	tst, ign = parseFile(devDatasets)
	
	#ws = trainPer(trainingDatasets, 1, 0.01)
	#test(devDatasets, ws)

	'''
	w = []
	for i in range(69):
		w.append(0.0)
	w[0] = 1
	test(devDatasets, w)
	test(testDatasets, w)
	'''
	fr = 5
	to = 11
	if len(argv) > 1:
		fr = int(argv[0])
		to = int(argv[1])

	#createTrees(trainingDatasets, testDatasets)

	#nMaxlr, nMaxAcc = naiveCrossVal()
	'''
	naive = naiveTrain(train, num+1, 1.5)
	naiveTest(tst, naive)
	'''

	#tes, ign = parseFile2(["moreData/bt100/treeOut.train"])
	#print(tes)
	
	#trainxor = [[-1, []], [1, [0,1.0],[1, [1,1.0]],[-1, [0,1.0], [1,1.0]]]]
	#tree = createTree(train, 10)
	#predicted = genLabels(tree, train)

	#Cross val
	#mode = 0
	accs =[]
	avAccs = 0
	mtext = ["Vanilla", "Dynamic", "Margin", "Averaged", "Aggressive", "1_SGDSVM", "2_Logistic_Regression", "4_Bagged_trees", "5_SVM_Over_Trees", "6_LogReg_over_trees", "3_Naive_Bayes"]
	for mode in range(fr,to):

		print("######################## Mode", mode+1, ":", mtext[mode], "###########################")
		print("########################################################################\n")
		
		if mode == 7:
			baggedTrees = bagTrees(train, 1000, 3)
			treeT, ign = parseFile(testDatasets)
			prvc = treeTest(baggedTrees, treeT)
		elif mode == 8 or mode == 9:
			if mode == 8:
				print("SVM over trees")
			else:
				print("LogReg over trees")
			print("Cross validation testing...")
			dbest = crossValAux(mode, 1)
			print("Development testing...")
			ep, ws, c = devTestOver(mode, dbest[0], dbest[1], dbest[3])
			print(dbest)
		elif mode == 10:
			print("Naive_Bayes")
			nMaxlr, nMaxAcc = naiveCrossVal()
			learnt = naiveTrain(train, num+1, nMaxlr)
			acc = naiveTest(tst, learnt)
			print("Final accuracy for Naive Bayes test = ", acc, " Best ")
		else:
			#Cross val
			print("Cross validation testing...")
			learnRate, marg = crossVal(mode, 1)
			#Testing on development set
			print("Development testing...")
			ep, ws, c = devTest(mode, learnRate, marg)

			print("-------------------------------------Final testing------------------------------------")
			
			#ws, ignore, c = trainPerTest(train, num, ep, learnRate, tst, mode, marg)
			if(mode == 2):
				print("Training on training set for ", ep, "epochs, learning rate", learnRate, "Updates made =",c, "Margin =", marg, "Mode = ", mode + 1)
			elif(mode == 4):
				print("Training on training set for ", ep, "epochs", "Updates made =",c, "Margin =", learnRate, "Mode = ", mode + 1)
			else:
				print("Training on training set for ", ep, "epochs, learning rate", learnRate, "Updates made =",c, "Mode = ", mode + 1)

			print("Testing on test set...")
			acc = 1
			if mode == 6:
				acc = test(testDatasets, ws, mtext[mode])
			else:
				acc = test(testDatasets, ws, mtext[mode])
			#test(devDatasets, ws)
			accs.append(acc)
			avAccs += acc
			print("--------------------------------------------------------------------------------------")
			print("\n\n\n")

	print(accs)
	#print(avAccs/20)

#Naive Bayes#########################################################
def naiveCrossVal():
	rCombs = [2, 1.5, 1, .5]
	trainFiles = ["CVSplits/training00.data", "CVSplits/training01.data", "CVSplits/training02.data", "CVSplits/training03.data", "CVSplits/training04.data"]
	ep = 10
	allAvsSds = []
	for r in rCombs:
		rccuracy = []
		for i in range(len(trainFiles)):
			tst = [trainFiles[i]]
			
			trainer = []
			for j in range(len(trainFiles)):
				if i != j:
					trainer.append(trainFiles[j])

			trainer, other = parseFile(trainer)

			
			print("Training for lambda = ", r, "Testing on ", tst)
			testSet, atM = parseFile(tst)
			ws, epochAccuracy = trainN(trainer, other+1, r, testSet)
			rccuracy.append(epochAccuracy)

		avs = 0
		for el in rccuracy:
			avs += el
		avs = avs/len(rccuracy)
		allAvsSds.append([avs,r])

	maxi = 0
	maxir = 2
	for val in allAvsSds:
		if val[0] > maxi:
			maxi = val[0]
			maxir = val[1]
		#if pri == 1:
			#print()

	print("Max mean accuracy was for ", maxir, " with accuracy ", maxi) 

	return maxir, maxi

def trainN(trainS, num, lam, testSet):
	naiveLearn = naiveTrain(trainS, num, lam)
	acc = naiveTest(testSet, naiveLearn)

	return naiveLearn, acc

def naiveTrain(trainS, num, lam):
	p = [0,0]
	aj0 = []
	aj1 = []
	bj0 = []
	bj1 = []

	la = float(lam)

	for i in range(num):
		aj0.append(la)
		aj1.append(la)
		bj0.append(la)
		bj1.append(la)

	for ex in trainS:
		lab = ex[0][0]
		if(ex[0][0] == 1):
			p[1] += 1
		else:
			p[0] += 1

		for i in range(num):
			aj0[i] += 1
			bj0[i] += 1

		for i in range(1, len(ex)):
			ind = ex[i][0]
			#print(ind, " ", num, "\n")
			if lab == 1:
				aj0[ind] -= 1
				aj1[ind] += 1
			else:
				bj0[ind] -= 1
				bj1[ind] += 1

	for i in range(num):
		aj = aj0[i] + aj1[i]
		if aj0[i] != 0:
			aj0[i] = math.log(aj0[i]/aj, 2)
			#print(aj0[i])
		if aj1[i] != 0:
			aj1[i] = math.log(aj1[i]/aj, 2)
		bj = bj0[i] + bj1[i]
		if bj0[i] != 0:
			bj0[i] = math.log(bj0[i]/bj, 2)
		if bj1[i] != 0:
			bj1[i] = math.log(bj1[i]/bj, 2)

	ps = p[0] + p[1]
	if p[0] != 0:
		p[0] = math.log(p[0],2)
	if p[1] != 0:
		p[1] = math.log(p[1],2)

	naiveLearn = [p, aj0, aj1, bj0, bj1]

	return naiveLearn

def naiveTest(tst, naive):
	#print("Stuff")
	p = naive[0]	
	aj0 = naive[1]
	aj1 = naive[2]
	bj0 = naive[3]
	bj1 = naive[4]

	predictions = []
	good = 0.0

	for ex in tst:
		finLab = 0
		oneto = []
		zeroto = []
		for i in range(len(aj0)):
			oneto.append(aj0[i])
			zeroto.append(bj0[i])

		for i in range(1, len(ex)):
			feat = ex[i]
			if feat[0] < len(aj0):
				oneto[feat[0]] = oneto[feat[0]] - aj0[feat[0]] + aj1[feat[0]]
				zeroto[feat[0]] = zeroto[feat[0]] - bj0[feat[0]] + bj1[feat[0]]

		oneprob = p[1]
		zeroprob = p[0]
		for i in range(len(oneto)):
			oneprob += oneto[i]
			zeroprob += zeroto[i]

		ans = 0
		#print(oneprob, " ", zeroprob)
		if oneprob > zeroprob:
			predictions.append(1)
			ans = 1
		else:
			predictions.append(-1)
			ans = -1

		if ans == ex[0][0]:
			good += 1

	acc = float(good/len(tst))
	print("Accuracy ", acc, " Good ", good)
	return acc

#SVM and logreg over trees###########################################
def createTrees(trainingDatasets, testDatasets):
	hyppar = [50,100,150, 200]

	for i in range(len(hyppar)):
		baggedTrees = bagTrees(train, 1000, hyppar[i])
		treeT, ign = parseFile(trainingDatasets)
		prvc = treeTest(baggedTrees, treeT)
		fnames = "moreData/bt" + str(hyppar[i]) +"/"
		treewrite(prvc, fnames)
		treeT, ign = parseFile(testDatasets)
		prvc = treeTest(baggedTrees, treeT)
		twrite(prvc, fnames+"treeOut.test")

def crossValAux(typ, pri):
	hyppar = [50,100,150, 200]

	bestHypparAll = []
	bestAcc = 0
	for i in range(len(hyppar)):
		fnames = "moreData/bt" + str(hyppar[i]) +"/"
		print("Cross validating for " + str(hyppar[i]) + " Folder " + fnames)
		bestHyppar = crossValOver(typ,pri,fnames)
		bestHyppar = bestHyppar + (fnames,)
		if bestHyppar[2] > bestAcc:
			bestAcc = bestHyppar[2]
			bestHypparAll = bestHyppar

	print("#####################################")
	print("End Best Acc for crossval for mode " +  str(typ + 1) + " = " + str(bestHypparAll))
	print("#####################################")

	return bestHypparAll

def crossValOver(typ, pri, folder):
	margins = [1.0, 0.1, 0.01]
	if typ == 8:
		margins = [10, 1, 0.1, 0.01, 0.001, 0.0001]
	else:
		margins = [0.1, 1, 10, 100, 1000, 10000]
	maxAv = 0
	bestMargin = 1.0
	bestRs = 1.0
	for margin in margins:
		print("\nFor margin =", margin)
		bestR, av = crossVal2Over(typ, pri, margin, folder)
		if(av > maxAv):
			maxAv = av
			bestMargin = margin
			bestRs = bestR
	return bestRs, bestMargin, maxAv

def crossVal2Over(typ, pri, margin, folder):
	rCombs = [1.0, 0.1, 0.01]
	if typ == 8:
		rCombs = [10, 1, 0.1, 0.01, 0.001, 0.0001]
	else:
		rCombs = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

	trainFiles = ["CVSplits/training00.data", "CVSplits/training01.data", "CVSplits/training02.data", "CVSplits/training03.data", "CVSplits/training04.data"]
	
	for i in range(len(trainFiles)):
		trainFiles[i] = folder + trainFiles[i]
	ep = 10
	allAvsSds = []
	for r in rCombs:
		rccuracy = []
		for i in range(len(trainFiles)):
			tst = [trainFiles[i]]
			
			trainer = []
			for j in range(len(trainFiles)):
				if i != j:
					trainer.append(trainFiles[j])

			trainer, other = parseFile2(trainer)

			if pri == 1:
				print("Training for epochs = ", ep, "learning rate = ", r, "Testing on ", tst)
			testSet, atM = parseFile2(tst)
			ws, epochAccuracy,c = trainPerTest(trainer, other, ep, r, testSet, typ, margin)
			rccuracy.append(epochAccuracy)

		aVs = []
		for i in range(len(rccuracy[0])):
			aVs.append(0.0)
			for j in range(len(rccuracy)):
				aVs[i] += rccuracy[j][i]
			aVs[i] = aVs[i]/len(rccuracy)
			#print("Average for", i+1, "epochs = ", aVs[i])

		sdin = []
		for i in range(len(rccuracy[0])):
			sdin.append(0.0)
			for j in range(len(rccuracy)):
				sdin[i] += (rccuracy[j][i] - aVs[i])**2
			sdin[i] = math.sqrt(sdin[i])
			#print("SD for", i+1, "epochs = ", sdin[i])

		ans = []

		for i in range(len(sdin)):
			t = []
			t.append(aVs[i]) 
			t.append(sdin[i])
			ans.append(t)
		allAvsSds.append(ans)

	maxAv = 0
	maxI = 0
	maxJ = 0
	for i in range(ep):
		if pri == 1:
			print("Epoch:", i+1)
		for j in range(len(rCombs)):
			if pri == 1:
				print("[" , rCombs[j], ",", round(allAvsSds[j][i][0], 4), " +-", round(allAvsSds[j][i][1], 4), "]", end =" ")
				#print("Hello", end = "")
			if(allAvsSds[j][i][0] > maxAv):
				maxAv = allAvsSds[j][i][0] 
				maxI = i
				maxJ = j
		if pri == 1:
			print()

	print("Max mean accuracy was for ", maxI + 1, "epochs, with learning rate", rCombs[maxJ], "[", round(allAvsSds[maxJ][maxI][0], 4), "+-", round(allAvsSds[maxJ][maxI][1], 4), "]")

	return rCombs[maxJ], maxAv

def devTestOver(typ, lRate, marg, folder):
	training, num = parseFile2([folder + "treeOut.train"])
	testing, ignore = parseFile2([folder + "treeOut.train"])
	devDatasets = [folder + "treeOut.train"]
	
	ws, epochAccuracy, c, wss = trainPerTestDev(training, num, 40, lRate, testing, typ, marg)

	accMax = 0
	epMax = 0
	for i in range(len(epochAccuracy)):
		#rint("Accuracy for epoch", i+1, "=", epochAccuracy[i])
		print(epochAccuracy[i])
		#test(devDatasets, wss[i])
		if(epochAccuracy[i] > accMax):
			accMax = epochAccuracy[i]
			epMax = i + 1

	print("Max accuracy was obtained with", epMax, "epochs, with accuracy = ", accMax)

	return epMax, wss[epMax-1], c
#####################################################################

def devTest(typ, lRate, marg):
	training, num = parseFile(["speeches.train.liblinear"])
	testing, ignore = parseFile(["speeches.train.liblinear"])
	devDatasets = ["speeches.test.liblinear"]
	
	ws, epochAccuracy, c, wss = trainPerTestDev(training, num, 40, lRate, testing, typ, marg)

	accMax = 0
	epMax = 0
	for i in range(len(epochAccuracy)):
		#rint("Accuracy for epoch", i+1, "=", epochAccuracy[i])
		print(epochAccuracy[i])
		#test(devDatasets, wss[i])
		if(epochAccuracy[i] > accMax):
			accMax = epochAccuracy[i]
			epMax = i + 1

	print("Max accuracy was obtained with", epMax, "epochs, with accuracy = ", accMax)

	return epMax, wss[epMax-1], c

def crossVal2(typ, pri, margin):
	rCombs = [1.0, 0.1, 0.01]
	trainFiles = ["CVSplits/training00.data", "CVSplits/training01.data", "CVSplits/training02.data", "CVSplits/training03.data", "CVSplits/training04.data"]
	ep = 10
	allAvsSds = []
	for r in rCombs:
		rccuracy = []
		for i in range(len(trainFiles)):
			tst = [trainFiles[i]]
			
			trainer = []
			for j in range(len(trainFiles)):
				if i != j:
					trainer.append(trainFiles[j])

			trainer, other = parseFile(trainer)

			if pri == 1:
				print("Training for epochs = ", ep, "learning rate = ", r, "Testing on ", tst)
			testSet, atM = parseFile(tst)
			ws, epochAccuracy,c = trainPerTest(trainer, other, ep, r, testSet, typ, margin)
			rccuracy.append(epochAccuracy)

		aVs = []
		for i in range(len(rccuracy[0])):
			aVs.append(0.0)
			for j in range(len(rccuracy)):
				aVs[i] += rccuracy[j][i]
			aVs[i] = aVs[i]/len(rccuracy)
			#print("Average for", i+1, "epochs = ", aVs[i])

		sdin = []
		for i in range(len(rccuracy[0])):
			sdin.append(0.0)
			for j in range(len(rccuracy)):
				sdin[i] += (rccuracy[j][i] - aVs[i])**2
			sdin[i] = math.sqrt(sdin[i])
			#print("SD for", i+1, "epochs = ", sdin[i])

		ans = []

		for i in range(len(sdin)):
			t = []
			t.append(aVs[i]) 
			t.append(sdin[i])
			ans.append(t)
		allAvsSds.append(ans)

	maxAv = 0
	maxI = 0
	maxJ = 0
	for i in range(ep):
		if pri == 1:
			print("Epoch:", i+1)
		for j in range(len(rCombs)):
			if pri == 1:
				print("[" , rCombs[j], ",", round(allAvsSds[j][i][0], 4), " +-", round(allAvsSds[j][i][1], 4), "]", end =" ")
				#print("Hello", end = "")
			if(allAvsSds[j][i][0] > maxAv):
				maxAv = allAvsSds[j][i][0] 
				maxI = i
				maxJ = j
		if pri == 1:
			print()

	print("Max mean accuracy was for ", maxI + 1, "epochs, with learning rate", rCombs[maxJ], "[", round(allAvsSds[maxJ][maxI][0], 4), "+-", round(allAvsSds[maxJ][maxI][1], 4), "]")

	return rCombs[maxJ], maxAv

def crossVal3(typ, pri, margin, rCombs):
	trainFiles = ["CVSplits/training00.data", "CVSplits/training01.data", "CVSplits/training02.data", "CVSplits/training03.data", "CVSplits/training04.data"]
	ep = 10
	allAvsSds = []
	for r in rCombs:
		rccuracy = []
		for i in range(len(trainFiles)):
			tst = [trainFiles[i]]
			
			trainer = []
			for j in range(len(trainFiles)):
				if i != j:
					trainer.append(trainFiles[j])

			trainer, other = parseFile(trainer)

			if pri == 1:
				print("Training for epochs = ", ep, "learning rate = ", r, "Testing on ", tst)
			testSet, atM = parseFile(tst)
			ws, epochAccuracy,c = trainPerTest(trainer, other, ep, r, testSet, typ, margin)
			rccuracy.append(epochAccuracy)

		aVs = []
		for i in range(len(rccuracy[0])):
			aVs.append(0.0)
			for j in range(len(rccuracy)):
				aVs[i] += rccuracy[j][i]
			aVs[i] = aVs[i]/len(rccuracy)
			#print("Average for", i+1, "epochs = ", aVs[i])

		sdin = []
		for i in range(len(rccuracy[0])):
			sdin.append(0.0)
			for j in range(len(rccuracy)):
				sdin[i] += (rccuracy[j][i] - aVs[i])**2
			sdin[i] = math.sqrt(sdin[i])
			#print("SD for", i+1, "epochs = ", sdin[i])

		ans = []

		for i in range(len(sdin)):
			t = []
			t.append(aVs[i]) 
			t.append(sdin[i])
			ans.append(t)
		allAvsSds.append(ans)

	maxAv = 0
	maxI = 0
	maxJ = 0
	for i in range(ep):
		if pri == 1:
			print("Epoch:", i+1)
		for j in range(len(rCombs)):
			if pri == 1:
				print("[" , rCombs[j], ",", round(allAvsSds[j][i][0], 4), " +-", round(allAvsSds[j][i][1], 4), "]", end =" ")
				#print("Hello", end = "")
			if(allAvsSds[j][i][0] > maxAv):
				maxAv = allAvsSds[j][i][0] 
				maxI = i
				maxJ = j
		if pri == 1:
			print()

	print("Max mean accuracy was for ", maxI + 1, "epochs, with learning rate", rCombs[maxJ], "[", round(allAvsSds[maxJ][maxI][0], 4), "+-", round(allAvsSds[maxJ][maxI][1], 4), "]")

	return rCombs[maxJ], maxAv

def crossVal(typ, pri):

	if typ == 2:
		margins = [1.0, 0.1, 0.01]
		maxAv = 0
		bestMargin = 1.0
		bestRs = 1.0
		for margin in margins:
			print("\nFor margin =", margin)
			bestR, av = crossVal2(typ, pri, margin)
			if(av > maxAv):
				maxAv = av
				bestMargin = margin
				bestRs = bestR
		return bestRs, bestMargin

	if typ == 5:
		margins = [10, 1, 0.1, 0.01, 0.001, 0.001]
		rCombs = [10, 1, 0.1, 0.01, 0.001, 0.001]
		maxAv = 0
		bestMargin = 1.0
		bestRs = 1.0
		for margin in margins:
			print("\nFor margin =", margin)
			bestR, av = crossVal3(typ, pri, margin, rCombs)
			if(av > maxAv):
				maxAv = av
				bestMargin = margin
				bestRs = bestR
		return bestRs, bestMargin

	if typ == 6:
		margins = [0.1 , 1, 10, 100, 1000, 10000]
		rCombs = [1, 0.1, 0.01, 0.001, 0.001, 0.0001]
		maxAv = 0
		bestMargin = 1.0
		bestRs = 1.0
		for margin in margins:
			print("\nFor margin =", margin)
			bestR, av = crossVal3(typ, pri, margin, rCombs)
			if(av > maxAv):
				maxAv = av
				bestMargin = margin
				bestRs = bestR
		return bestRs, bestMargin

	rCombs = [1.0, 0.1, 0.01]
	trainFiles = ["CVSplits/training00.data", "CVSplits/training01.data", "CVSplits/training02.data", "CVSplits/training03.data", "CVSplits/training04.data"]
	ep = 10
	allAvsSds = []
	for r in rCombs:
		rccuracy = []
		for i in range(len(trainFiles)):
			tst = [trainFiles[i]]
			
			trainer = []
			for j in range(len(trainFiles)):
				if i != j:
					trainer.append(trainFiles[j])

			trainer, other = parseFile(trainer)

			if pri == 1:
				print("Training for epochs = ", ep, "learning rate = ", r, "Testing on ", tst)
			testSet, atM = parseFile(tst)
			ws, epochAccuracy,c = trainPerTest(trainer, other, ep, r, testSet, typ, r)
			rccuracy.append(epochAccuracy)

		aVs = []
		for i in range(len(rccuracy[0])):
			aVs.append(0.0)
			for j in range(len(rccuracy)):
				aVs[i] += rccuracy[j][i]
			aVs[i] = aVs[i]/len(rccuracy)
			#print("Average for", i+1, "epochs = ", aVs[i])

		sdin = []
		for i in range(len(rccuracy[0])):
			sdin.append(0.0)
			for j in range(len(rccuracy)):
				sdin[i] += (rccuracy[j][i] - aVs[i])**2
			sdin[i] = math.sqrt(sdin[i])
			#print("SD for", i+1, "epochs = ", sdin[i])

		ans = []

		for i in range(len(sdin)):
			t = []
			t.append(aVs[i]) 
			t.append(sdin[i])
			ans.append(t)
		allAvsSds.append(ans)

	maxAv = 0
	maxI = 0
	maxJ = 0
	for i in range(ep):
		if pri == 1:
			print("Epoch:", i+1)
		for j in range(len(rCombs)):
			if pri == 1:
				print("[", rCombs[j], ",", round(allAvsSds[j][i][0], 4), " +-", round(allAvsSds[j][i][1], 4), "]", end =" ")
			if(allAvsSds[j][i][0] > maxAv):
				maxAv = allAvsSds[j][i][0] 
				maxI = i
				maxJ = j
		if pri == 1:
			print()

	print("Max mean accuracy was for ", maxI + 1, "epochs, with learning rate", rCombs[maxJ], "[", round(allAvsSds[maxJ][maxI][0], 4), "+-", round(allAvsSds[maxJ][maxI][1], 4), "]")

	if(type == 4):
		return rCombs[maxJ], rCombs[maxJ]

	return rCombs[maxJ], 0

def test(filenames, ws, name):
	plines, s = parseFile(filenames)

	file = open("data/speeches.test.liblinear", 'r')

	goods = 0

	finished = {}

	for i in range(len(plines)):
		ex = plines[i]
		label = ex[0][0]
		result = ws[0]
		for i in range(1, len(ex)-1):
			if(ex[i][0] < len(ws)):
				result += ws[ex[i][0]]*ex[i][1]
		#################################################################################
		'''line = int(file.readline())
		if line not in finished:
			if result > 0:
				finished[line] = "1"
			else:
				finished[line] = "0"'''
		#################################################################################
		if(label*result > 0):
			goods += 1
	print("Result =", result, " label = ", label, " = ", goods)

	#print(finished)
	writeCSV(finished, name)

	print ("Final count = ", goods, "/", len(plines), " or accuracy = ", goods/len(plines))
	return goods/len(plines)

def testLog(filenames, ws, name):
	plines, s = parseFile(filenames)

	file = open("data/speeches.test.liblinear", 'r')

	goods = 0

	finished = {}

	for i in range(len(plines)):
		ex = plines[i]
		label = ex[0][0]
		result = ws[0]
		for i in range(1, len(ex)-1):
			if(ex[i][0] < len(ws)):
				result += ws[ex[i][0]]*ex[i][1]
		#################################################################################
		'''line = int(file.readline())
		if line not in finished:
			if result > 0:
				finished[line] = "1"
			else:
				finished[line] = "0"'''
		#################################################################################
		predLabel = 1
		if 1.0/(1.0 + math.exp(-result)) < 0.5:
			predLabel = -1
		if(label*predLabel > 0):
			goods += 1
	print("Result =", result, " label = ", label, " = ", goods)

	#print(finished)
	writeCSV(finished, name)

	print ("Final count = ", goods, "/", len(plines), " or accuracy = ", goods/len(plines))
	return goods/len(plines)

def testLog2(filenames, ws, name):
	plines, s = parseFile(filenames)

	file = open("data/speeches.test.liblinear", 'r')

	goods = 0

	finished = {}

	for i in range(len(plines)):
		ex = plines[i]
		label = ex[0][0]
		result = ws[0]
		for i in range(1, len(ex)-1):
			if(ex[i][0] < len(ws)):
				result += ws[ex[i][0]]*ex[i][1]
		#################################################################################
		'''line = int(file.readline())
		if line not in finished:
			if result > 0:
				finished[line] = "1"
			else:
				finished[line] = "0"'''
		#################################################################################
		result = -result
		if(label*result > 0):
			goods += 1
	print("Result =", result, " label = ", label, " = ", goods)

	#print(finished)
	writeCSV(finished, name)

	print ("Final count = ", goods, "/", len(plines), " or accuracy = ", goods/len(plines))
	return goods/len(plines)

def writeCSV(res, filename):
	file = open("output/" + filename + ".csv", 'w')
	file.write("Id,Prediction\n")
	for key in res:
		file.write(str(key) + "," + res[key] + "\n")
	file.close()

def testFile(plines, ws):

	goods = 0

	for ex in plines:
		label = ex[0][0]
		result = ws[0]
		for i in range(1, len(ex)-1):
			#print(ex[i][0])
			if(ex[i][0] < len(ws)):
				result += ws[ex[i][0]]*ex[i][1]
		good = 0
		if(label*result > 0):
			good = 1
			goods += 1
		#print("Result =", result, " label = ", label, " = ", good)
	#print(goods/len(plines))
	#print ("\t\tFinal count = ", goods, "/", len(plines), " or accuracy = ", goods/len(plines))
	return goods/len(plines)

def testFileLog2(plines, ws):

	goods = 0

	for ex in plines:
		label = ex[0][0]
		result = ws[0]
		for i in range(1, len(ex)-1):
			#print(ex[i][0])
			if(ex[i][0] < len(ws)):
				result += ws[ex[i][0]]*ex[i][1]
		result = -result
		good = 0
		if(label*result > 0):
			good = 1
			goods += 1
		#print("Result =", result, " label = ", label, " = ", good)
	print(goods/len(plines))
	#print ("\t\tFinal count = ", goods, "/", len(plines), " or accuracy = ", goods/len(plines))
	return goods/len(plines)

def testFileLog(plines, ws):

	goods = 0

	for ex in plines:
		label = ex[0][0]
		result = ws[0]
		for i in range(1, len(ex)-1):
			#print(ex[i][0])
			if(ex[i][0] < len(ws)):
				result += ws[ex[i][0]]*ex[i][1]
		good = 0
		predLabel = 1
		if result > 400 and 1.0/(1.0 + math.exp(-result)) < 0.5:
			predLabel = -1
		if(label*predLabel > 0):
			goods += 1
			good = 1
		#print(label + ", " + predLabel + "||" + goods)
		#print("Result =", result, " label = ", label, " = ", good)
	#print(goods/len(plines))

	#print ("\t\tFinal count = ", goods, "/", len(plines), " or accuracy = ", goods/len(plines))
	return goods/len(plines)

def trainPer(trainingDatasets, epochs, lRate):
	tExamples, atM = parseFile(trainingDatasets)

	ws = []
	for i in range(atM):
		ws.append(random.uniform(-0.01,0.01))

	#print(ws)
	for i in range(epochs):
		ws = epoch(ws,tExamples,lRate)
	#print(ws)

	return ws

def trainPerTestDev(train, atM, epochs, lRate, test, typ, margin):
	wss = []
	ws = []
	aws = []
	for i in range(atM):
		if(typ == 6):
			ws.append(0.0)
		else:
			ws.append(random.uniform(-0.01,0.01))
		aws.append(0)


	#print(ws)
	c = 0
	epAccs = []
	avAcc = 0
	count = 0
	for i in range(epochs):
		random.shuffle(train)
		if typ == 0:
			c, ws = epoch(c,ws,train,lRate)
		if typ == 1:
			c,ws, count = epoch1(c,ws, train, lRate, count)
		if typ == 2:
			c,ws, count = epoch2(c,ws, train, lRate, count, margin)
		if typ == 3:
			c,ws, aws = epoch3(c,ws,train,lRate, aws)
		if typ == 4:
			#print(ws)
			c,ws = epoch4(c,ws, train, margin)
			#print("\tEpoch number ", i)
		if typ == 5 or typ == 8:
			c,ws, count = epoch5(c,ws, train, lRate, count, margin)
		if typ == 6 or typ == 9:
			c,ws, count = epoch6(c,ws, train, lRate, count, margin)
		count += 1

		acc = 0
		if typ == 6:
			acc = testFile(test,ws)
		else:	
			acc = testFile(test, ws)

		wsp = lis(ws)
		if typ == 3:
			acc = testFile(test, aws)
			awsp = lis(aws)
			wss.append(awsp)
		else:
			wss.append(wsp)
		avAcc += acc
		#print(round(acc, 4), end = " ")
		epAccs.append(acc)
	#print(ws)
	avAcc = avAcc/epochs
	#print()
	#print("Average Accuracy = ", avAcc)

	if typ == 3:
		return aws, epAccs, c, wss
	return ws, epAccs, c, wss

def lis(r):
	s = []
	for el in r:
		s.append(el)
	return s

def trainPerTest(train, atM, epochs, lRate, test, typ, margin):
	ws = []
	aws = []
	for i in range(atM):
		if(typ == 6):
			ws.append(0.0)
		else:
			ws.append(random.uniform(-0.01,0.01))
		aws.append(0)

	#print(train)
	#print(ws)
	c = 0
	epAccs = []
	avAcc = 0
	count = 0
	for i in range(epochs):
		random.shuffle(train)
		if typ == 0:
			c, ws = epoch(c,ws,train,lRate)
		if typ == 1:
			c,ws, count = epoch1(c,ws, train, lRate, count)
		if typ == 2:
			c,ws, count = epoch2(c,ws, train, lRate, count, margin)
		if typ == 3:
			c,ws, aws = epoch3(c,ws,train,lRate, aws)
		if typ == 4:
			#print(ws)
			c,ws = epoch4(c,ws, train, margin)
			#print("\tEpoch number ", i)
		if typ == 5 or typ == 8:
			c,ws, count = epoch5(c,ws, train, lRate, count, margin)
		if typ == 6 or typ == 9:
			c,ws, count = epoch6(c,ws, train, lRate, count, margin)
		count += 1
		#acc = testFile(test, ws)

		acc = 0
		if typ == 6 or typ == 9:
			acc = testFile(test,ws)
			print(acc)
			for counter in range(10):
				print(str(ws[counter]) + " ", end = '')
			print("")
		else:	
			acc = testFile(test, ws)

		if typ == 3:
			acc = testFile(test, aws)
		avAcc += acc
		#print(round(acc, 4), end = " ")
		epAccs.append(acc)
	#print(ws)
	avAcc = avAcc/epochs
	#print()
	#print("Average Accuracy = ", avAcc)

	if typ == 3:
		return aws, epAccs, c
	return ws, epAccs, c

def epoch(c,ws, trainE, lRate):
	for ex in trainE:
		label = ex[0][0]
		result = ws[0]
		for i in range(1, len(ex)-1):
			result += ws[ex[i][0]]*ex[i][1]
		if(label*result < 0):
			for i in range(1, len(ex)-1):
				ws[ex[i][0]] = ws[ex[i][0]] + lRate*label*ex[i][1]
			ws[0] = ws[0] + lRate*label
			c += 1
	return c,ws

def epoch3(c,ws, trainE, lRate, aws):
	for ex in trainE:
		label = ex[0][0]
		result = ws[0]
		for i in range(1, len(ex)-1):
			result += ws[ex[i][0]]*ex[i][1]
		if(label*result < 0):
			for i in range(1, len(ex)-1):
				ws[ex[i][0]] = ws[ex[i][0]] + lRate*label*ex[i][1]
			ws[0] = ws[0] + lRate*label
			c += 1
		for i in range(len(aws)):
			aws[i] = aws[i] + ws[i]
	return c,ws, aws

def epoch1(c,ws, trainE, baselRate, count):
	for ex in trainE:
		lRate = baselRate/(1+count)
		label = ex[0][0]
		result = ws[0]
		for i in range(1, len(ex)-1):
			result += ws[ex[i][0]]*ex[i][1]
		if(label*result < 0):
			for i in range(1, len(ex)-1):
				ws[ex[i][0]] = ws[ex[i][0]] + lRate*label*ex[i][1]
			ws[0] = ws[0] + lRate*label
			c += 1
		#count += 1
	return c,ws, count 

def epoch2(c,ws, trainE, baselRate, count, margin):
	for ex in trainE:
		lRate = baselRate/(1+count)
		label = ex[0][0]
		result = ws[0]
		for i in range(1, len(ex)-1):
			result += ws[ex[i][0]]*ex[i][1]
		if(label*result < margin):
			for i in range(1, len(ex)-1):
				ws[ex[i][0]] = ws[ex[i][0]] + lRate*label*ex[i][1]
			ws[0] = ws[0] + lRate*label
			c += 1
		#count += 1
	return c,ws, count 

def epoch5(c,ws, trainE, lRate, count, cArg):
	print("Loading...")
	lr = 1 - lRate
	for ex in trainE:
		#print(c)
		label = ex[0][0]
		result = ws[0]

		for i in range(1, len(ex)-1):
			result += ws[ex[i][0]]*ex[i][1]
		
		for i in range(len(ws)):
			ws[i] = ws[i]*lr
			c += 1
		
		ywx = label * result
		#print(ywx)

		if(ywx <= 1.0):
			for i in range(1, len(ex)-1):
				ws[ex[i][0]] += lRate*cArg*label*ex[i][1]
				#ws[ex[i][0]] = ws[ex[i][0]]*lr + lRate*cArg*label*ex[i][1]
			ws[0] += lRate*label*cArg
			#ws[0] = ws[0]*lr + lRate*label*cArg
			
		#count += 1
	return c,ws, count 

def epoch6(c,ws, trainE, lRate, count, cArg):
	print("Loading...")
	lr = 1 - lRate
	threshold = 500
	exp4001 = math.exp(threshold) + 1
	for ex in trainE:
		#print(c)
		label = ex[0][0]
		result = ws[0]

		for i in range(1, len(ex)-1):
			result += ws[ex[i][0]]*ex[i][1]
		

		#ws[0] = 2/cArg*ws[0] - ((label*1*math.exp(-label*w[0]*1))/(1+ math.exp(-label*w[0]*1)))
		
		for i in range(len(ws)):
			ws[i] = (1-2*lRate/cArg)*ws[i]
			c += 1
		
		#ywx = label * result
		#print(ywx)

		for i in range(1, len(ex)-1):
			#ws[ex[i][0]] += lRate*cArg*label*ex[i][1]
			#ws[ex[i][0]] -= ((label*ex[i][1]*math.exp(-label*ws[ex[i][0]]*ex[i][1]))/(1+ math.exp(-label*ws[ex[i][0]]*ex[i][1])))
			#print(label*ex[i][1]*ws[ex[i][0]])
			tempval = label*ex[i][1]*ws[ex[i][0]]
			#print(tempval)
			if(tempval < threshold):
				ws[ex[i][0]] = ws[ex[i][0]] + lRate*label*ex[i][1]/(math.exp(tempval) + 1)
			else:
				ws[ex[i][0]] = ws[ex[i][0]] + lRate*label*ex[i][1]/(exp4001)
		#ws[0] += lRate*label*cArg
		#ws[0] -= ((label*1*math.exp(-label*ws[0]*1))/(1+ math.exp(-label*ws[0]*1)))
		#print(label*ws[0])
		tempval = label*ws[0]
		if(tempval< threshold):
			ws[0] = ws[0] + lRate*label/(math.exp(tempval)+1)
		else:
			ws[0] = ws[0] + lRate*label/(exp4001)
		#count += 1
	return c,ws, count 

def epoch4(c,ws, trainE, margin):
	for ex in trainE:
		label = ex[0][0]
		result = ws[0]
		xtx = 1
		for i in range(1, len(ex)-1):
			result += ws[ex[i][0]]*ex[i][1]
			xtx += ex[i][1]**2
		if(label*result <= margin):

			num = margin - label*result
			denom = xtx + 1
			lRate = num/denom
			#print(ws)
			#print(lRate, " ", num , "/ ", denom, " ", result, " ", ws[0])

			for i in range(1, len(ex)-1):
				ws[ex[i][0]] = ws[ex[i][0]] + lRate*label*ex[i][1]
			ws[0] = ws[0] + lRate*label
			c += 1
	return c,ws

def sum2lists(lista, listb):
	for i in range(len(lista)):
		lista[i] = lista[i] + listb[i]
	return lista

def parseFile(filenames):
	lCount = 0
	#print(file.read())
	attMax = 0
	virginLines = []
	lines = []
	for filename in filenames:
		file = open("data/" + filename,"r")
		for line in file:
			lCount+=1
			virginLines.append(line)
			line = line.rstrip("\n").lower()
			line = re.sub('  ', ' ', line)
			lSplit = line.split(" ")
			
			lSplit2 = []
			for w in lSplit:
				if w:
					w = w.split(":")
					#print([w])
					t = [int(w[0])]
					if len(w) > 1:
						t.append(float(w[1]))
						if int(w[0]) > attMax:
							attMax = int(w[0])
					#print(t)
					lSplit2.append(t)

			lines.append(lSplit2)
		#print(wordCount)
	#print(lCount)
	return lines, attMax

def parseFile2(filenames):
	lCount = 0
	#print(file.read())
	attMax = 0
	virginLines = []
	lines = []
	for filename in filenames:
		file = open(filename,"r")
		for line in file:
			lCount+=1
			virginLines.append(line)
			line = line.rstrip("\n").lower()
			line = re.sub('  ', ' ', line)
			lSplit = line.split(" ")
			
			lSplit2 = []
			for w in lSplit:
				if w:
					w = w.split(":")
					#print([w])
					t = [int(w[0])]
					if len(w) > 1:
						t.append(float(w[1]))
						if int(w[0]) > attMax:
							attMax = int(w[0])
					#print(t)
					lSplit2.append(t)

			lines.append(lSplit2)
		#print(wordCount)
	#print(lCount)
	return lines, attMax

#Writing bt####################################################################
def treewrite(vc, n):
	twrite(vc,n+"treeOut.train")

	splits = []

	for i in range(5):
		splits.append([])
	
	for ex in vc:
		ran = randint(0, 4)
		if len(splits[ran]) <= len(vc)/5 + 1:
			splits[ran].append(ex)
		else:
			while 1:
				ran = randint(0, 4)
				if len(splits[ran]) <= len(vc)/5 + 1:
					splits[ran].append(ex)
					break

	for i in range(len(splits)):
		twrite(splits[i], n+"CVSplits/training0" + str(i) +".data")

def twrite(vc, name):
	file = open(name, 'w')

	for j in range(len(vc)):
		ex = vc[j]
		file.write(str(ex[0]))
		for i in range(1, len(ex)):
			file.write(" "+str(ex[i][0])+":"+str(ex[i][1]))
		if j != len(vc) - 1:
			file.write("\n")

#Bagged trees##################################################################
def bagTrees(vec,num,depth):
	trees = []
	for i in range(num):
		print("Training tree " +str(i))
		sampVec = []
		for j in range(100):
			sampVec.append(vec[randint(0,len(vec)-1)])
		trees.append(createTree(sampVec, depth))
	return trees

def treeTest(trees, test):
	predictVec = []
	for tree in trees:
		print("Loading...")
		pred = genLabels(tree, test)
		#print(pred)
		predictVec.append(pred)

	finPreds = []
	newVec = []
	for i in range(len(predictVec[0])):
		newVec.append([])
		newVec[i].append(test[i][0][0])
		print("Loading...")
		ones= 0
		mones = 0
		for j in range(len(predictVec)):
			if predictVec[j][i] == 1:
				newVec[i].append([j, 1.0])
				ones += 1
			else:
				mones += 1
		if ones > mones:
			finPreds.append(1)
		else:
			finPreds.append(-1)

	#print(finPreds)

	good = 0.0
	total = float(len(finPreds))
	for i in range(len(finPreds)):
		#print(test[i][0][0])
		if finPreds[i] == test[i][0][0]:
			good += 1.0

	print("Bagged trees accuracy: " + str(good/total))
	#print(newVec)
	return newVec

#Creating trees################################################################
def createTree(realVecs, maxDepth):

	dic, maxfea = dicfy(realVecs)
	#print(dic)

	order = calculateOr(dic, maxfea)
	
	root = Tree()
	root = recurseTree(root, dic, 2, 3, maxDepth, 0, order)
	
	return root

def recurseTree(node, realVecs, div, ids, maxdepth, curdepth, order):
	if curdepth >= maxdepth or curdepth >= len(order):
		return maxLabel(realVecs)

	node.name = order[curdepth]

	for i in range(div):
		subsetAeV = []
		node.children.append(Tree(order[curdepth]))
		for element in realVecs:
			if order[curdepth] in element and element[order[curdepth]] == i:
				subsetAeV.append(element)
		if len(subsetAeV) == 0:
			node.children[i] = maxLabel(realVecs)
		else:
			node.children[i] = (recurseTree(node.children[i], subsetAeV, div, ids, maxdepth, curdepth + 1, order))

	return node

def maxLabel(realVecs):
	one = 0
	zero = 0
	for element in realVecs:
		if element[0] == 1:
			one += 1
		else:
			zero += 1

	if one > zero:
		return "1"
	else:
		return "0"

def calculateOr(dic, maxfea):
	e = []
	mine = 1
	for i in range(1, maxfea + 1):
		ones = float(0.0)
		onesone = float(0.0)
		zeros = float(0.0)
		zerosone = float(0.0)
		for ex in dic:
			if i in ex:
				ones += 1
				if ex[0] == 1:
					onesone += 1
			else:
				zeros += 1
				if ex[0] == 1:
					zerosone += 1

		oneszero = ones - onesone
		zeroszero = zeros - zerosone

		eonesone = 0
		if onesone != 0 :
			eonesone = -onesone/ones*math.log(onesone/ones,2)
		eoneszero = 0
		if oneszero != 0:
			eoneszero = - oneszero/ones*math.log(oneszero/ones,2)
		ezerosone = 0
		if zerosone != 0:
			ezerosone = -zerosone/zeros*math.log(zerosone/zeros,2)
		ezeroszero = 0
		if zeroszero != 0:
			ezeroszero = - zeroszero/zeros*math.log(zeroszero/zeros, 2)

		#print(ones)

		entropy1 = ones/(zeros + ones) *(eonesone + eoneszero) 
		entropy0 = zeros/(ones+zeros)*(ezerosone + ezeroszero)
		entropy = entropy1 + entropy0
		e.append(entropy)
		if entropy < mine:
			mine = entropy
	print("Min-entropy = " + str(mine))

	sortede = sorted(range(len(e)), key=lambda k: e[k])

	av = 0
	for i in range(len(sortede)):
		sortede[i] += 1
		av += e[i]
	av = av/len(e)
	#print(e)
	#print(sortede)
	print("Average entropy = " + str(av))
	return sortede

def dicfy(vector):
	dics = []
	maxfea = 0
	for v in vector:
		dic = {}
		dic[0] = v[0][0]
		for i in range(1, len(v)):
			ent = v[i]
			dic[ent[0]] = ent[1]
			if ent[0] > maxfea:
				maxfea = ent[0]
		dics.append(dic)
	return dics, maxfea

#Test trees####################################################################

def genLabels(decTree, realVecs):
	realVecs, maxfea = dicfy(realVecs)
	#print(realVecs)
	predicted = []
	for el in realVecs:
		#print("el = " + str(el))
		label = recurseGenLables(decTree, el)
		if label == '1':
			predicted.append(1)
		else:
			predicted.append(-1)

	'''
	right = 0
	for i in range(len(realVecs)):
		if predicted[i] == realVecs[i][0]:
			right += 1
	accuracy = float(float(right)/float(len(realVecs)))

	#print(predicted)
	#print("Accuracy = " + str(accuracy))
	'''
	return predicted

def recurseGenLables(decTree, vector):
	if decTree.name in vector:
		if isinstance(decTree.children[int(vector[decTree.name])], str):
			return  decTree.children[int(vector[decTree.name])]
		else:
			return recurseGenLables(decTree.children[int(vector[decTree.name])], vector)
	else:
		if isinstance(decTree.children[0], str):
			return  decTree.children[0]
		else:
			return recurseGenLables(decTree.children[0], vector)

###############################################################################

class Tree(object):
    "Generic tree node."
    def __init__(self, name='root', children=None):
        self.name = name
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)
    def __repr__(self):
        return self.name
    def add_child(self, node):
        assert isinstance(node, Tree)
        self.children.append(node)

if __name__ == "__main__":
   main(sys.argv[1:])