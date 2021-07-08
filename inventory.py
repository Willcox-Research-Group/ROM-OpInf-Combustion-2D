# inventory.py
"""List existing files and folders in the base folder."""

import os
import re
import glob

import config


def _sglob(x):
    """Sorted glob()."""
    return sorted(glob.glog(x))


# Regular expresssions --------------------------------------------------------
_s = "    "                                 # Indentation space.
_trnpat = re.compile(fr".*{config.TRN_PREFIX}(\d+)")
_dimpat = re.compile(fr".*{config.DIM_PREFIX}(\d+)")
_regpat = re.compile(fr".*{config.ROM_PREFIX}_{config.REG_PREFIX}([\d_]+)\.h5")


def _get_trn(foldername):
    result = _trnpat.findall(foldername)
    return int(result[0]) if result else None


def _get_dim(foldername):
    result = _dimpat.findall(foldername)
    return int(result[0]) if result else None


def _get_regs(filename):
    result = _regpat.findall(filename)
    return [int(reg) for reg in result[0].split('_')] if result else None


# Main routines ---------------------------------------------------------------

def print_rom_folder(folder):
    num_modes = _get_dim(folder)
    print(f"{_s*2}ROM dimension = {num_modes} ({os.path.basename(folder)}/)")
    prefix = f"{config.ROM_PREFIX}_{config.REG_PREFIX}"
    romfiles = []
    for dfile in _sglob(os.path.join(folder, "*.h5")):
        basename = os.path.basename(dfile)
        if basename.startswith(prefix):
            romfiles.append(dfile)
        else:
            print(f"{_s*3}{basename}")
    for dfile in romfiles:
        regs = _get_regs(dfile)
        print(f"{_s*3}* Trained ROM with λ1 = {regs[0]}, λ2 = {regs[1]}"
              f" ({os.path.basename(dfile)})")


def print_trainsize_folder(folder):
    trainsize = _get_trn(folder)
    print(f"    Train size = {trainsize} ({os.path.basename(folder)}/)")
    for dfile in _sglob(os.path.join(folder, "*.h5")):
        basename = os.path.basename(dfile)
        if basename == config.SCALED_DATA_FILE:
            print(f"{_s*2}* Scaled data ({basename})")
        elif basename == config.BASIS_FILE:
            print(f"{_s*2}* POD basis ({basename})")
        elif basename == config.PROJECTED_DATA_FILE:
            print(f"{_s*2}* Projected data ({basename})")
        else:
            print(f"{_s*2}* {basename}")
    for romfolder in _sglob(os.path.join(folder, config.DIM_PREFIX + '*')):
        print_rom_folder(romfolder)


def main():
    print(f"BASE FOLDER: ({config.BASE_FOLDER}/)")
    for dfile in _sglob(os.path.join(config.BASE_FOLDER, "*.h5")):
        basename = os.path.basename(dfile)
        if basename == config.GEMS_DATA_FILE:
            print(f"{_s}* GEMS data file ({basename})")
        elif basename == config.FEATURES_FILE:
            print(f"{_s}* Features file ({basename})")
        else:
            print(f"{_s}* {basename}")
    for folder in _sglob(os.path.join(config.BASE_FOLDER,
                                      config.TRN_PREFIX + '*')):
        print_trainsize_folder(folder)


if __name__ == "__main__":
    # Set up command line argument parsing.
    import argparse
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=formatter)
    parser.usage = f""" python3 {__file__} [--help]"""

    # Parse "arguments" and call main routine.
    args = parser.parse_args()
    main()
