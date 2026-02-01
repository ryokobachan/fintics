"""Ultra-minimal test to verify pytest is working."""

def test_basic_math():
    """Test that basic math works."""
    assert 1 + 1 == 2
    assert 2 * 3 == 6
    assert 10 / 2 == 5

def test_string_operations():
    """Test that string operations work."""
    text = "hello world"
    assert text.upper() == "HELLO WORLD"
    assert text.split() == ["hello", "world"]

def test_list_operations():
    """Test that list operations work."""
    numbers = [1, 2, 3, 4, 5]
    assert len(numbers) == 5
    assert sum(numbers) == 15
    assert max(numbers) == 5

if __name__ == "__main__":
    print("Running basic tests...")
    test_basic_math()
    test_string_operations()
    test_list_operations()
    print("All tests passed!")
