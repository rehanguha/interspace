# calculate the Euclidean distance between two vectors
def euclidean(vector_1, vector_2):
    distance = 0.0
    if len(vector_1) == len(vector_2):
    	for i in range(len(vector_1)):
    		distance += (vector_1[i] - vector_2[i])**2
    	return (distance)**0.5
    else:
        raise Exception("Vector lengths mismatch")


def distance(vector_1, vector_2):
    distances = {}
    try:
        distances['Euclidean'] = euclidean(vector_1, vector_2)
        return distances
    except:
        print("Vector lengths mismatch")