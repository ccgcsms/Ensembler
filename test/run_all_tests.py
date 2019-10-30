import os, sys
import unittest

#import importlib

if __name__ == "__main__":

    #include path:
    print("PWD", os.getenv("PWD"))

    raw_path = os.getenv("PWD")
    print(raw_path.split("/Ensembler"))
    include_path = raw_path.split("/Ensembler")[0]
    sys.path.append(include_path)
    print(include_path)

    #FILE MANAGMENT
    test_root_dir = os.path.dirname(__file__)
    print("TEST ROOT DIR: " + test_root_dir)

    ##gather all test_files
    test_files = []
    for dir in os.walk(test_root_dir):
        test_files.extend([dir[0]+"/"+path for path in dir[2] if( path.startswith("test") and path.endswith(".py") and not "test_run_all_tests" in path)])
    if(len(test_files) == 0):
        raise IOError("Could not find any test in : ", test_root_dir)

    ##get module import paths - there should be a function for that around
    modules = []
    for test_module in test_files:
        module_name =  "Ensembler"+test_module.replace(os.path.dirname(test_root_dir), "").replace("/", ".").replace(".py", "")
        if(module_name.startswith(".")): module_name = module_name[1:]
        modules.append(module_name)

    #LOAD TESTS
    print("LOAD TESTS")
    suite = unittest.TestSuite()
    test_loader = unittest.TestLoader()
    first = True

    for test_module in modules:
        print("Loading:\t", test_module)
        if("conveyor" in test_module):
            continue
        print("importing: ",test_module)
        imported_test_module = __import__(test_module, globals(), locals(), ['suite'])
        if(first):
            suite = test_loader.loadTestsFromModule(imported_test_module)
            first = False
        else:
            tmp = test_loader.loadTestsFromModule(imported_test_module)
            suite.addTest(tmp)

    #RUN TESTS
    print("RUN TESTS")
    try:
        print("TEST SUIT TESTS: ", suite.countTestCases())
        test_runner = unittest.TextTestRunner(verbosity=5)
        test_runner.run(suite)
        exit(0)
    except Exception as err:
        print("Test did not finish successfully!\n\t"+"\n\t".join(err.args))
        exit(1)
