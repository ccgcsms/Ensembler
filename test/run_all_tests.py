import os, sys
import unittest

#import importlib

if __name__ == "__main__":
    file = __file__
    root_dir = os.path.dirname(os.getenv("PWD"))

    test_files = []

    print("PWD", os.getenv("PWD"))
    print("THisFile: ", file)
    print("ROOT DIR: "+root_dir)
    sys.path.append(root_dir)
    sys.path.append("/root")

    #FILE MANAGMENT
    ##gather all test_files
    for dir in os.walk(root_dir):
        test_files.extend([dir[0]+"/"+path for path in dir[2] if( path.startswith("test") and path.endswith(".py") and not "test_run_all_tests" in path)])

    ##get module import paths
    modules = []
    for test_module in test_files:
        module_name =  test_module.replace(os.path.dirname(root_dir), "").replace("/", ".").replace(".py", "")
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
