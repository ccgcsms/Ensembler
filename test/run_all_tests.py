import os, sys
import unittest
import importlib

if __name__ == "__main__":
    print("RUNNING ALL TESTS\n")
    print("root_dir_path:\t", os.getcwd())
    print("root_dir_cont: ", os.listdir(os.getcwd()))
    
    test_dir_path = os.path.dirname(__file__)
    test_dir_path = "/test" if(test_dir_path == "test") else test_dir_path
    print("test_dir_path:\t", test_dir_path)

    root_dir_path = os.path.dirname(test_dir_path)
    root_dir_path = "/" if(root_dir_path == "") else root_dir_path
    print("root_dir_path:\t", root_dir_path)
    print("root_dir_cont: ", os.listdir(root_dir_path))
    root_dir_name = os.path.basename(root_dir_path)

    os.chdir(root_dir_path)
    sys.path.append(root_dir_path)

    print(root_dir_path)
    print("cwd ", os.listdir(os.getcwd()))
    print(".\t", os.listdir(root_dir_path))
    print("./test/\t", os.listdir(test_dir_path))

    #gather all test_files
    test_files = []
    for dir in os.walk(test_dir_path):
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
    #modules = ["test.test_potential1D", "test.test_potentialND", "test.test_system"]
    for test_file in modules:
        print("\tTry loading: ", root_dir_name+"." + test_file, "\n")
        mod = importlib.import_module(root_dir_name+"." + test_file)
        suite.addTest(unittest.defaultTestLoader.loadTestsFromName(root_dir_name+"." +test_file))

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
