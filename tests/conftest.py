# The legacy demo scripts (test.py, randWalk.py, test_interactive.py) drive the
# old terrapin.py model at import time and are not pytest tests; ignore them
# during collection. They will be removed when the old machinery is deleted.
collect_ignore = ["test.py", "randWalk.py", "test_interactive.py"]
