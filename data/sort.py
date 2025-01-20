# script i wrote in 2m to sort stuff
from glob import iglob
from os import makedirs, path
from shutil import move

PATTERN = "*.png"

def get(it):
    try:
        return next(it)
    except StopIteration:
        return None

while filename := get(iglob(PATTERN)):
    clazz, name = filename.split("-", 1)
    makedirs(clazz)

    for file in iglob(f"{clazz}-{PATTERN}"):
        new = path.join(clazz, "/", name)
        move(file, new)
        print(f"{file} -> {str(new)}")
