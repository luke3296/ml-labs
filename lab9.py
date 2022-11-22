import lab7_mnist
data = lab7_mnist.load_data("./../data/mnist.pkl/mnist.pkl")

class filter:
    #a matrix as a list of lists
    stepSize=1
    def __init__(self, k):
        self.kernel = k
        outputs=[]
    #assume the list of lists is not ragged
    def feature(self, inputMatrix):
        idim = len(inputMatrix) - (len(self.kernel)-self.stepSize)
        jdim= len(inputMatrix[0]) - (len(self.kernel[0])-self.stepSize)

        for i in range(idim):
            for j in range(jdim):
                print("i ", i, " j ", j)



input = [[1,1,1,0,1],
        [1,1,1,1,1],
        [0,0,0,0,0],
        [1,1,1,1,1],
        [1,1,0,1,1]]

pooling_mat = [[1,0,-1],
               [1,0,-1],
               [1,0,-1]]

f = filter(pooling_mat)

f.feature(input)
