import os
import sys


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)


from test_competing_food_firms import main


if __name__ == "__main__":
    if "--competition-steps" not in sys.argv:
        sys.argv.extend(["--competition-steps", "1600"])
    if "--segment-boundaries" not in sys.argv:
        sys.argv.extend(["--segment-boundaries", "600", "1000", "1500"])
    main()
