import requests
import os
import pickle
import sys
import time

from .helpers_misc import get_time_passed, save_star_coords

__all__ = ['get_data']

# ------------------------------------------------------------------------------
def http_get(path, params=None, print_progress=True):
    # based on http://www.tng-project.org/data/docs/api/
    # with code added to read in chuncks and print progress bar
    headers = {"api-key":"cbeb6728f96d7eced79c0cac712e8f15"}
    # make http get request to path
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically

    if 'content-disposition' in r.headers:
        filename = r.headers['content-disposition'].split("filename=")[1]
        # --------
        if print_progress:
            # set up to track progress
            dl = 0
            total_length = int(r.headers.get('content-length'))

            print("\nDownloading %s" % filename)
        # --------
        # now open the file
        with open(filename, 'wb') as f:
            # --------
            for data in r.iter_content(chunk_size=4096):
                f.write(data)
                # --------
                # print update
                # --------
                if print_progress:
                    dl += len(data)
                    #print(dl, total_length)
                    done = int(50 * dl / total_length)
                    sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)) )
                    sys.stdout.flush()
            # --------
        if print_progress: print('\nAll done.\n')
        return filename # return the filename string

    return r
# ------------------------------------------------------------------------------
def get_data(run_name, z, haloIDs, outdir, print_progress=True, readme=None):
    """

    This function reads in and saves the halo info and cutout.

    Inputs
    ------
    * run_name: str: either Illustris-1 or TNG-100
    * z: float: redshift of the halo
    * haloIDs: list of int: list of haloIDs
    * outdir: str: path to where the data should be stored.
    * print_progress (optional): bool
    * readme (optional): readme object or None

    """
    base_url = 'http://www.tng-project.org/api/'
    # --------------------------------------------------------------------------
    if run_name != 'Illustris-1' and run_name != 'TNG-100':
        raise ValueError('run_name must be either Illustris-1 or TNG-100. Input: %s' % run_name)
    # set up
    current_dir = os.getcwd()
    # --------------------------------------------------------------------------
    for haloID in haloIDs:
        start_time0 = time.time()
        # make the folder where to output things
        out_folder = '%s/%s_halo%s_z%s' % (outdir, run_name, haloID, z)
        if readme is not None:
            update = 'Getting data for haloID = %s\n' % (haloID)
            update += 'Saving data in %s' % (out_folder.split(outdir)[-1])
            readme.update(to_write=update)
        # make the out folder
        os.system('mkdir -p %s' % (out_folder))
        os.chdir(out_folder)
        # --------------------------------------------------------------------------
        # get halo info
        # --------------------------------------------------------------------------
        # based on https://github.com/HongyuLi2016/illustris-tools/blob/master/data/illustris-get_cutout.py
        url = "%s/%s/snapshots/z=%s/subhalos/%s" % (base_url, run_name, z, haloID)
        info_url = http_get(url, print_progress=print_progress)['meta']['info']
        info = http_get(info_url, print_progress=print_progress)
        # set up the dictionary
        info_dict = {}
        for key in info.keys():
            info_dict[key] = info[key]
        # save the dictionary
        filename = 'info.dat'
        with open(filename, 'wb') as f:
            pickle.dump(info_dict, f)
        # update readme
        if readme is not None:
            readme.update(to_write='Saved %s' % (filename))
        # --------------------------------------------------------------------------
        # get the cutout
        # --------------------------------------------------------------------------
        url += '/cutout.hdf5'
        start_time = time.time()
        filename_cutout = http_get(url, print_progress=print_progress)
        # update readme
        if readme is not None:
            update = 'Saved %s; filesize = %.2f GB\n' % (filename_cutout, os.path.getsize(filename_cutout)/1e9)
            update += '## Time taken: %s\n'%get_time_passed(start_time)
            readme.update(to_write=update)

        # --------------------------------------------------------------------------
        # read the cutout and save only the star coordinates; remove full cutout.
        # --------------------------------------------------------------------------
        start_time = time.time()
        # save star coords
        filename_coords = save_star_coords(fname_cutout=filename_cutout, z=z)
        # now remove the cutout file
        os.remove(filename_cutout)
        # update readme
        if readme is not None:
            update = 'Saved star coords in %s; filesize = %.2f GB\n' % (filename_coords,
                                                                        os.path.getsize(filename_coords)/1e9)
            update += 'Removed %s\n' % filename_cutout
            update += '## Time taken: %s\n'%get_time_passed(start_time)
            update += '-----------------------------------------------\n'
            readme.update(to_write=update)
    # change back to starting directory.
    os.chdir(current_dir)
    if readme is not None:
        update = 'All done.\n## Time taken: %s\n'%get_time_passed(start_time0)
        readme.update(to_write=update)
    return
