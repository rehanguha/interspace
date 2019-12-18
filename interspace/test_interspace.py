import interspace
import pytest

def test_validate_var_type():
    assert interspace._validate_var_type(21, int) == 21
    with pytest.raises(ValueError, match=r".*int.*"):
        interspace._validate_var_type(21.1, int)
    with pytest.raises(ValueError, match=r".*int.*"):
        interspace._validate_var_type("string", int)
  
    assert interspace._validate_var_type(21.1, float) == 21.1
    with pytest.raises(ValueError, match=r".*float.*"):
        interspace._validate_var_type(21, float)
    with pytest.raises(ValueError, match=r".*float.*"):
        interspace._validate_var_type("string", float)    

    assert interspace._validate_var_type("string", str) == "string"
    with pytest.raises(ValueError, match=r".*str.*"):
        interspace._validate_var_type(21.1, str)
    with pytest.raises(ValueError, match=r".*str.*"):
        interspace._validate_var_type(21, str)

def test_validate_vector():
    assert interspace._validate_vector([21], dtype=int) == 21
    with pytest.raises(ValueError):
        interspace._validate_vector([21, 'a'], dtype=int)
    with pytest.raises(ValueError):
        interspace._validate_vector([1, [2]], dtype=int)

def test_validate_weights():
    assert interspace._validate_weights(21) == 21
    with pytest.raises(ValueError):
        interspace._validate_weights(-1)
    with pytest.raises(ValueError):
        interspace._validate_weights("a")