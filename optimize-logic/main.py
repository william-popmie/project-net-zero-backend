from function_spec import FunctionSpec
from optimizer import optimize_function

# Hardcoded test input — stands in for what your friend will pass
spec = FunctionSpec(
    function_name="is_prime",
    module_path="input-folder/mock_repo/math_utils.py",
    function_source="""def is_prime(n):
    if n < 2:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True""",
    test_source="""def test_is_prime():
    assert is_prime(2) == True
    assert is_prime(4) == False
    assert is_prime(17) == True""",
)

result = optimize_function(spec)
print(result)
