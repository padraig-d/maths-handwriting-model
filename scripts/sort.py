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
    clazz = filename.split("-", 1)[0]
    makedirs(clazz)

    for file in iglob(f"{clazz}-{PATTERN}"):
        name = file.split("-", 1)[1]
        new = path.join(f"{clazz}/", name)
        move(file, new)
        print(f"{file} -> {str(new)}")
