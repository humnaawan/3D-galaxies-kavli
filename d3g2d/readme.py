import time
import os
import sys
import datetime

__all__ = ['readme']

# ------------------------------------------------------------------------------
class readme(object):
    """

    This class takes care of creating, writing and updating a readme file.

    """
    def __init__(self, outdir, readme_tag, first_update):
        self.outdir = outdir
        self.readme_tag = readme_tag
        self.first_update = first_update

    def create(self):
        self.filename = 'readme%s'%self.readme_tag
        # see if there are any readmes that are already saved
        saved_readmes = os.listdir(self.outdir)
        saved_readmes = [f for f in saved_readmes \
                         if (any([f.endswith('.txt')]) \
                             and any([f.__contains__(self.filename)])
                             )]
        num_file = 0  # number of readme files already saved
        # loop over the saved readmes to see if any matches
        # the tag intended for this run
        for file in saved_readmes:
            if file.__contains__('%s_'%self.filename):
                # found a file. account for in counter.
                temp = file.split('.txt')[0]
                num_file = max(num_file,
                               int(temp.split('%s_'%self.filename)[1]))
            else:
                num_file = 1
        # update the readme name with the filenumber tag.
        self.readme_tag += '_%s'%(num_file+1)
        # store the filename
        self.filename = 'readme%s.txt'%(self.readme_tag)
        # write the first update
        readme = open('%s/%s'%(self.outdir, self.filename), 'w')
        readme.write(self.first_update)
        readme.close()

    def update(self, to_write):
        readme = open('%s/%s'%(self.outdir, self.filename), 'a')
        readme.write('\n%s'%to_write)
        readme.close()

    def run(self):
        self.create()
