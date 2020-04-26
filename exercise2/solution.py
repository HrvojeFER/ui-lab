import sys

from shells import shell_factory

verbose = 'verbose'

argv = sys.argv[1:]
if verbose not in argv:
    shell = shell_factory(*argv)
else:
    shell = shell_factory(*argv[:argv.index(verbose)], verbose=True)

shell.run()
