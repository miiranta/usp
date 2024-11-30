#Lucas Miranda Mendonça Rezende
#12542838

#==================================================================================================

#libs
import matplotlib.pyplot as plt
from math import sqrt
from math import exp
from math import pi

#Vars
label_dic = {0:"Circulo", 1:"Quadrado", 2:"Losango"}

#==================================================================================================

#Plota o dataset
#[x1, x2, classe]
#classe = 0:Circulo, 1:Quadrado, 2:Losango
def plot_dataset(dataset):
    color_codes = {0:'red',1:'green',2:'blue'}
    marker_codes = {0:'o',1:'s',2:'D'}
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Dataset")
    plt.xticks(x)
    plt.yticks(y)
    plt.xlim([0,10])
    plt.ylim([0,10])
    plt.grid()
    
    x1 = [row[0] for row in dataset]
    x2 = [row[1] for row in dataset]
    mk = [row[2] for row in dataset]
    
    for i,j in enumerate(x1):
        plt.scatter(x1[i] , x2[i], color = color_codes.get(mk[i], 'black'), marker = marker_codes.get(mk[i], '.'))

    plt.show()

#Plota as fronteiras de decisão
def plot_boundries(model):
    precision = 50 #Vai com calma! é o(n^2).
    
    color_codes = {0:'red',1:'green',2:'blue'}
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Boundries")
    plt.xticks(x)
    plt.yticks(y)
    plt.xlim([0,10])
    plt.ylim([0,10])
    plt.grid()
    
    testset = []
    for i in range(precision+1):
        for j in range(precision+1):
            x1 = (i/precision)*10
            x2 = (j/precision)*10
            label = predict(model, [x1, x2])
            testset.append([x1, x2, label])
    
    x1 = [row[0] for row in testset]
    x2 = [row[1] for row in testset]
    mk = [row[2] for row in testset]
    
    for i,j in enumerate(x1):
        plt.scatter(x1[i] , x2[i], color = color_codes.get(mk[i], 'black'), marker = "$.$")

    plt.show()

#==================================================================================================

# Split the dataset by class values, returns a dictionary
def separate_by_class(dataset):
	separated = dict()
	for i in range(len(dataset)):
		vector = dataset[i]
		class_value = vector[-1]
		if (class_value not in separated):
			separated[class_value] = list()
		separated[class_value].append(vector)
	return separated

# Calculate the mean of a list of numbers
def mean(numbers):
	return sum(numbers)/float(len(numbers))

# Calculate the standard deviation of a list of numbers
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
	return sqrt(variance)

# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(dataset):
	summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
	del(summaries[-1])
	return summaries

# Split dataset by class then calculate statistics for each row
def summarize_by_class(dataset):
	separated = separate_by_class(dataset)
	summaries = dict()
	for class_value, rows in separated.items():
		summaries[class_value] = summarize_dataset(rows)
	return summaries

# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent

# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
	total_rows = sum([summaries[label][0][2] for label in summaries])
	probabilities = dict()
	for class_value, class_summaries in summaries.items():
		probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
		for i in range(len(class_summaries)):
			mean, stdev, _ = class_summaries[i]
			probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
	return probabilities

# Predict the class for a given row
def predict(summaries, row):
	probabilities = calculate_class_probabilities(summaries, row)
	best_label, best_prob = None, -1
	for class_value, probability in probabilities.items():
		if best_label is None or probability > best_prob:
			best_prob = probability
			best_label = class_value
	return best_label

#==================================================================================================

#Executa um predict para cada ponto do dataset fornecido
def run_predicts(model, points):
    for point in points:
        new_point = [point[0], point[1]]
        print("Point: ", new_point, "Label: ", label_dic[predict(model, new_point)])
    
#==================================================================================================

#Define o dataset
dataset = [ [0, 10, 1],
            [2, 9, 1],
            [4, 10, 1],
            [4, 9, 1],
            [4, 7, 1],
            [8, 10, 1],
            
            [1, 1, 0],
            [0, 4, 0],
            [2, 5, 0],
            [4, 3, 0],
            [8, 1, 0],
            
            [10, 3, 2],
            [8, 3, 2],
            [8, 4, 2],
            [9, 5, 2],
            [6, 5, 2],
            [9, 9, 2] 
        ]

#Define conjunto de teste 1 e 2
teste_1 = [ [3, 3],
            [1, 2],
            [7, 7],
            [8, 6],
            [4, 8],
            [6, 10]
        ]   

teste_2 = [ [2, 2],
            [6, 1],
            [1, 6],
            [3, 9],
            [9, 6],
            [3, 3],
            [10, 0],
            [1, 6],
            [0, 9],
            [4, 6],
            [5, 5],
            [10, 3]
        ]

#Plota o dataset
plot_dataset(dataset)

#Cria um modelo
model = summarize_by_class(dataset)

#Plota as fronteiras de decisão
plot_boundries(model)

#Define um novo exemplo e tenta prever a classe
exemplo = [5,5]
label = predict(model, exemplo)
print('Point:  %s Label: %s' % (exemplo, label_dic[label]))

#Executa um predict para cada ponto dos datasets de teste e treinamento
print("\nTeste 1: ")
run_predicts(model, teste_1)
print("\nTeste 2:")
run_predicts(model, teste_2)
print("\nTreinamento:")
run_predicts(model, dataset)

#==================================================================================================

#Bibliografia   
    #Implementação do algoritmo Naive Bayes --------------
    #Referência: https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
    
    #Matplotlib ------------------------------------------
    #Referência: https://www.w3schools.com/python/matplotlib_labels.asp
    #Referência: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html