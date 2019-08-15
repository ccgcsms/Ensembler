import os, sys
import unittest
import importlib

if __name__ == "__main__":
    test_dir = os.path.dirname(__file__)
    root_dir = test_dir+"/.."

    os.chdir(root_dir)
    sys.path.append(os.listdir(root_dir))
    print(".\t", os.listdir(root_dir))
    print("./test/\t", os.listdir(root_dir+"/test"))

    #gather all test_files
    test_files = []
    for dir in os.walk(test_dir):
        test_files.extend([dir[0]+"/"+path for path in dir[2] if( path.startswith("test") and path.endswith(".py"))])
    print(test_files)

    #do tests:
    suite = unittest.TestSuite()
    modules = []
    print("CHECKING Tests")
    for test_file in test_files:
        module_name = test_file[test_file.index("test"):].replace("/", ".").replace(".py", "")
        modules.append(module_name)

    print("LOADING Tests")
    for test_file in modules:
        print("\tTry loading: ", test_file, "\n")
        mod = importlib.import_module("Ensembler."+test_file, package=root_dir)
        suite.addTest(unittest.defaultTestLoader.loadTestsFromName(test_file))

        """
        try:
            # If the module defines a suite() function, call it to get the suite.
            mod = __import__("Ensembler."+test_file, globals(), locals(), ['suite'])
            suitefn = getattr(mod, 'suite')
            suite.addTest(suitefn())
        except (ImportError, AttributeError):
            # else, just load all the test cases from the module.
            suite.addTest(unittest.defaultTestLoader.loadTestsFromName(test_file))
        """
    try:
        print("RUNNING Tests")
        unittest.TextTestRunner().run(suite)
        print("All test did finish successfully!")
        exit(0)
    except:
        print("Test did not finish successfully!")
        exit(1)
