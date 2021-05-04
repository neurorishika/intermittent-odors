from subprocess import call
import sys

for i in [59428,13674,84932,72957,85036]:
    call(["python","onlyLNs.py",sys.argv[1],str(i)])
