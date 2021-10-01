# inventory.py
"""List existing files and folders in the base folder."""

import os
import re
import glob
import json
import shutil

import config


_s = "    "                                 # Indentation space.


def _sglob(x):
    """Sorted glob()."""
    return sorted(glob.glob(x))


# Regular expresssions --------------------------------------------------------

_trnpat = re.compile(fr".*{config.TRN_PREFIX}(\d+)")
_dimpat = re.compile(fr".*{config.DIM_PREFIX}(\d+)")


def _get_trn(foldername):
    """Extract the training size from the name of the folder."""
    result = _trnpat.findall(foldername)
    return int(result[0]) if result else None


def _get_dim(foldername):
    """Extract the ROM dimension from the name of the folder."""
    result = _dimpat.findall(foldername)
    return int(result[0]) if result else None


# Main routines ---------------------------------------------------------------

def print_rom_index(folder, clean=False):
    """Print and optionally reconcile a ROM index (config.ROM_INDEX_FILE).

    Parameters
    ----------
    folder : str
        Path to the folder to be examined, e.g,. BASE_FOLDER/k1000/.
    clean : bool
        If True, remove ROM files that are not listed in the ROM index and
        remove ROM index entries that do not have corresponding ROM files.
    """
    # Load the ROM index if it exists.
    rom_json = os.path.join(folder, config.ROM_INDEX_FILE)
    if not os.path.isfile(rom_json):
        rom_data = {}
    else:
        with open(rom_json, 'r') as infile:
            rom_data = json.load(infile)
        print(f"{_s*2}* ROM index ({os.path.basename(rom_json)})")

    if clean:
        # Remove items from the index if the corresponding file is missing.
        missing_folders, missing_files = [], []
        for rlabel in rom_data:
            rfolder = os.path.join(folder, rlabel)
            if not os.path.isdir(rfolder) or len(rom_data[rlabel]) == 0:
                missing_folders.append(rlabel)
            else:
                for filename in rom_data[rlabel]:
                    rom_file = os.path.join(rfolder, filename)
                    if not os.path.isfile(rom_file):
                        missing_files.append((rlabel, filename))
        for rlabel in missing_folders:
            rom_data.pop(rlabel)
        for rlabel, filename in missing_files:
            rom_data[rlabel].pop(filename)
        with open(rom_json, 'w') as outfile:
            json.dump(rom_data, outfile, indent=4)

        # Remove any ROM files that aren't in the index.
        for romfolder in _sglob(os.path.join(folder, "r???")):
            if not os.path.isdir(romfolder):
                continue
            rlabel = os.path.basename(romfolder)
            if rlabel not in rom_data:
                shutil.rmtree(romfolder)
            else:
                for h5file in _sglob(os.path.join(romfolder, '*.h5')):
                    if os.path.basename(h5file) not in rom_data[rlabel]:
                        os.remove(os.path.join(romfolder, h5file))
                if not any(os.scandir(romfolder)):
                    os.rmdir(romfolder)

    for rlabel in rom_data:
        rfolder = os.path.join(folder, rlabel)
        num_modes = _get_dim(rlabel)
        print(f"{_s*2}ROM dimension = {num_modes} ({rlabel}/)")
        for filename, regs in rom_data[rlabel].items():
            print(f"{_s*3}* Trained ROM with {config.REGSTR(regs)}")


def print_trainsize_folder(folder, clean=False):
    """Print a folder grouping data for one choice of training size (k)."""
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
    print_rom_index(folder, clean=clean)


def main(clean=False):
    """Display (and clean up) file storage starting from BASE_FOLDER."""
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
        print_trainsize_folder(folder, clean=clean)


if __name__ == "__main__":
    # Set up command line argument parsing.
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.usage = f""" python3 {__file__} --help
    python3 {__file__} [--clean]"""

    parser.add_argument("--clean", action="store_true",
                        help="delete ROM files that are not listed in the "
                             "directory and remove directory entries where "
                             "the corresponding ROM file is missing")

    # Parse "arguments" and call main routine.
    args = parser.parse_args()
    main(clean=args.clean)
