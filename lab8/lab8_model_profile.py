import lab7_mnist_loader
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from  sklearn.metrics import ConfusionMatrixDisplay , accuracy_score


data = lab7_mnist_loader.load_data("./mnist.pkl/mnist.pkl")

train_set, validation_set, test_set = data

train_set_imgs, train_set_lables = train_set
test_set_imgs, test_set_lables = test_set

random_seed = 1

#takes a string path/filename.sav and a model to save
def save_model(path, model):
    pickle.dump(model, open(path, 'wb'))

def load_model(path):
    return pickle.load(open(path, 'rb'))

#if  done with logistic and sgd the model predicts only 1's? when the model has more than 1 hidden layer
classifier = MLPClassifier(hidden_layer_sizes=(10,), max_iter=100,activation='logistic',solver='sgd',random_state=random_seed)
classifier.fit(train_set_imgs, train_set_lables)
pridictions = classifier.predict(test_set_imgs)
ascore = accuracy_score(test_set_lables,pridictions)
# save the model
save_model("MLP_mnist", classifier);
#load the model 
loaded_classifier = load_model("./MLP_mnist");
pridictions = loaded_classifier.predict(test_set_imgs)
ascore = accuracy_score(test_set_lables,pridictions)
print("the accuracy was ", ascore)
#alpha - strength of L2 regularization term
#coefs_ list of shape (n_layers-1) where each element 
#corrosponds to the weight matrix corrosponding to 
#layer i


activations =  [ 'logistic', 'tanh', 'relu' ]
solver = [ 'sgd', 'adam']
deep_layers = [(10,), (10,10,10) ,(300,) , (300,300,300)]
itterations = [100,400]

best_accuracy=0;
for i in range(len(activations)):
    for j in range(len(solver)):
        for k in range(len(deep_layers)):
            for l in range(len(itterations)):
                classifier = MLPClassifier(hidden_layer_sizes=deep_layers[k], max_iter=itterations[l], activation=activations[i], solver=solver[j],
                                   random_state=random_seed)
                classifier.fit(train_set_imgs, train_set_lables)
                pridictions = classifier.predict(test_set_imgs)
                ascore = accuracy_score(test_set_lables, pridictions)
                if ascore > best_accuracy:
                    best_accuracy = ascore;
                    save_model("bestModel.sav", classifier)
                    print("saw best accuracy " ,ascore," with values ", deep_layers[k], itterations[l], activations[i], solver[j])
        #disp = ConfusionMatrixDisplay.from_predictions(test_set_lables, pridictions)

        #disp.figure_.suptitle("Confusion Matrix " + activations[i] + " " +solver[j])
        #plt.savefig("cf_matrix.png")
        #plt.show()

#input model: sklearn MLP,  labels: list of dimension 1 x num_instances containing the class varible, instances: list of size  num_instances x num_features
def get_missclassified(model,lables,instances):
    pridictions=model.predict(instanceses)
    missclassified=[]
    for i in range(len(pridictions)):
        if(pridictions[i] != lables[i]):
            missclassified.append(instances[i])
    return missclassified

best_model = load_model("./bestModel.sav")

missclassifiedInstances = get_missclassified(best_model, test_set_lables,test_set_imgs );

for inst in missclassifiedInstances:
    print(best_model.predict_proba(inst))

# returns the avrage of the top k accuracy probabilities
#def top_k_accuracy(k, model, tests):
    