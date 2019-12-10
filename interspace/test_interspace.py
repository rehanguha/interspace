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
    pass


def test_validate_weights():
    pass