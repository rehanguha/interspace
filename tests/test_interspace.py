import sys
sys.path.append('../src/')
import interspace

def test_euclidean():
    assert  interspace.euclidean([1212,1],[12,12]) == 1200.050415607611
    

def test_minkowski():
    assert interspace.minkowski([2,3,4], [4,5,6]) == -6.0
    assert interspace.minkowski([2,3,4], [4,5,6], p=2) == 3.4641016151377544