from gpmlx.main import function


def test_function():
    assert function(2) == 4
    assert function(0) == 0
    assert function(-3) == 9
