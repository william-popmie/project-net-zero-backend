from dataclasses import dataclass

@dataclass
class FunctionSpec:
    function_name: str
    module_path: str       # absolute path to the source file
    function_source: str   # raw source string of just the function
    test_source: str       # raw source string of generated test stubs
