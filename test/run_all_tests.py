import os, sys
import unittest


if __name__ == "__main__":
    root_dir = os.path.dirname(__file__)
    sys.path.append(root_dir.replace("/test", ""))
    print(root_dir.replace("/test", ""))
    print(os.listdir(root_dir))
    print(os.listdir(root_dir+"/.."))

    #gather all test_files
    test_files = []
    for dir in os.walk(root_dir):
        test_files.extend([dir[0]+"/"+path for path in dir[2] if( path.startswith("test") and path.endswith(".py"))])
    print(test_files)

    #do tests:
    suite = unittest.TestSuite()
    modules = []
    print("CHECKING Tests")
    for test in test_files:
        module_name = test[test.index("test"):].replace("/", ".").replace(".py", "")
        #module_name = test.replace(os.path.dirname(root_dir)+".", "").replace("/", ".").replace(".py", "")
        modules.append(module_name)

    print("LOADING Tests")
    for test_file in modules:
        print("\tTry loading: ", test_file, "\n")
        try:
            print(test_file)
            # If the module defines a suite() function, call it to get the suite.
            mod = __import__(test_file, globals(), locals(), ['suite'])
            suitefn = getattr(mod, 'suite')
            suite.addTest(suitefn())
        except (ImportError, AttributeError):
            # else, just load all the test cases from the module.
            suite.addTest(unittest.defaultTestLoader.loadTestsFromName(test_file))

    try:
        print("RUNNING Tests")
        unittest.TextTestRunner().run(suite)
        print("All test did finish successfully!")
        exit(0)
    except:
        print("Test did not finish successfully!")
        exit(1)
