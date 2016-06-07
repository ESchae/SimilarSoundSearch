# -*- coding: utf-8 -*-
"""
Copyright 2016 University of Freiburg
Elke Schaechtele <elke.schaechtele@web.de>

This file contains the main function for the similarity search for sounds.
"""
from __future__ import print_function
from soundbase import SoundBase
from search import Search
import os
from subprocess import call
from time import sleep
import random
input = getattr(__builtins__, 'raw_input', input)  # outsmart checkstyle


def test_audiofile_validity(filename):
    """ Helperfunction for testing whether an audiofile is valid.

    To be valid an audiofile must exist and have one of the supported
    file formats (flac, mp3, ogg, wav).

    Parameters
    ----------
        audiofile (str): path to the audiofile

    Returns
    -------
        None if the audiofile is valid, otherwise an exception will be thrown


    Examples
    --------
    >>> test_audiofile_validity('Testfiles/sine300.wav')
    >>> test_audiofile_validity('Testfiles/test.au')
    Traceback (most recent call last):
     ...
    IOError: Fileformat not supported.
    Supported audioformats: flac, mp3, oog, wav
    >>> test_audiofile_validity('doesnotexist.mp3')
    Traceback (most recent call last):
     ...
    IOError: File doesnotexist.mp3 does not exist
    """
    exts = ['.flac', '.mp3', '.ogg', '.wav']
    if not os.path.isfile(filename):
        e = "File %s does not exist" % filename
        raise IOError(e)
    elif not os.path.splitext(filename)[-1] in exts:
        e = 'Fileformat not supported.\n'
        e += 'Supported audioformats: flac, mp3, oog, wav'
        raise IOError(e)
    return None


def display_query(query, random_mode=False, listening=False):
    """ This function prints the given query , if wished in bold. """
    # go to beginning of line 4
    print("\033[4;0H")
    if random_mode:
        print("\033[1;32mYour random query: \033[0m", end='')
    else:
        print("\033[1;32mYour query: \033[0m", end='')
    if listening:
        # print bold and cyan
        print("\033[1;36m" + query + "\033[0m")
    else:
        print(query)


def display_result(i, n, path, distance, listening=False):
    line = 5 + i
    print("\033[%d;0H" % line)
    if listening:
        # print bold and cyan
        print("\033[1;36mNN %d: %s with distance %.2f\033[0m"
              % (i, path, distance))
        print("\033[%d;30H" % (n + 10))
    else:
        print("NN %d: %s with distance %.2f" % (i, path, distance))


if __name__ == "__main__":
    # console arguments handling
    import itertools
    queries = itertools.cycle([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    import argparse
    parser = argparse.ArgumentParser(
        description='A very helpful description...',
        epilog='...  have fun!')
    parser.add_argument('S', type=str,
                        help="""path to a sql database for sounds (aka
                                soundbase); can be build with the help
                                of the SoundBase class""")
    parser.add_argument('-q', '--q', type=str,
                        help='a query audiofile')
    parser.add_argument('-n', type=int,
                        help='number of results to be returned', default=14)
    parser.add_argument('-o', type=str,
                        help='print results additionally to this file')
    args = parser.parse_args()

    # ------------------------ preliminaries --------------------------- #

    # check if audiofile is valid if a query is given
    if args.q:
        test_audiofile_validity(args.q)

    # check if soundbase exists
    if not os.path.isfile(args.S):
        raise IOError("SoundBase %s does not exist" % args.S)

    # --------------- begin of interactive communication ------------------- #

    # print header
    print("\033[H\033[2J")  # erase display
    print("\033[1;32mWelcome to SimilarSoundsSearch!\033[0m")
    print()

    # print database
    print("\033[1;32mYour soundbase: \033[0m", end='')
    database = args.S
    db = SoundBase(database)
    max_rows = db.num_rows()
    print(database)

    s = Search(database, n=args.n)

    # get query
    if not args.q:
        random_mode = True
        query_id = random.randint(1, max_rows)
        query_id = queries.next()
        query = db.get_row(query_id)['path']
    else:
        random_mode = False
        query = args.q

    # print query
    display_query(query, random_mode=random_mode, listening=False)

    # print result
    result = s.query(query)
    print("\033[1;32mResult: \033[0m")
    s.print_result(result)

    if args.o:
        # write result to file
        with open(args.o, 'a') as f:
            f.write("Query: %s\n" % query)
            f.write("Result: \n")
            s.print_result(result, output=f)
            f.write("\n")

    print("\033[s")  # save cursor position
    while True:
        print("\033[u\033[0J")  # restore cursor position
        # go back to bottom bottom line
        a = input("Press n(ew query) | l(isten to any sound) | e(xit) ")
        if a == 'l':  # listen to the sounds
            query = query.replace(' ', '\ ')  # secure blank in filename
            while True:
                # restore cursor position and clear screen under it
                line = 6 + args.n + 1
                print("\033[%d;0H\033[0J" % line)
                print('Press any number (0 = query, 1-x = result in list)')
                print('or press f(inished listening) to go back\n')
                x = input('Which one do you want to listen to? ')
                try:
                    if x == 'f':
                        break
                    elif int(x) == 0:  # query sound should be played
                        # print query in bold
                        print("Press Ctrl+c to stop playing")
                        display_query(query, random_mode=random_mode,
                                      listening=True)
                        # play query
                        c = 'play %s -q -V0' % query
                    elif int(x) > len(result[1][0]) or int(x) < 0:
                        # x > num results or negative
                        print('The number you specified is not in the list')
                        sleep(1)
                        continue
                    else:
                        ids = result[1][0]
                        dist = result[0][0][int(x)-1]
                        row_id = ids[int(x)-1]
                        path = db.get_row(row_id+1)['path'].replace(' ', '\ ')
                        c = 'play %s -q -V0' % path
                        # print played sound in bold
                        print("Press Ctrl+c to stop playing")
                        display_result(int(x), args.n, path, dist,
                                       listening=True)
                    try:
                        call([c], shell=True)
                        # recover old display
                        if int(x) == 0:
                            display_query(query, random_mode=random_mode,
                                          listening=False)
                        else:
                            display_result(int(x), args.n, path, dist,
                                           listening=False)
                    except KeyboardInterrupt:  # workaround to stop sox playing
                        if int(x) == 0:
                            display_query(query, random_mode=random_mode,
                                          listening=False)
                        else:
                            display_result(int(x), args.n, path, dist,
                                           listening=False)
                        pass
                except:
                    print("You did not specify a valid number")
                    sleep(1)
                    continue
        elif a == 'n':
            try:
                query = input("New query file or r(andom): ")
                if query == 'r':  # randomly select new query
                    # query_id = random.randint(1, 10)
                    query_id = queries.next()
                    query = db.get_row(query_id)['path']
                    random_mode = True
                # go to line and erase screen under it
                print("\033[5;0H\033[J", end='')
                # print new query
                display_query(query, random_mode=random_mode, listening=False)
                # print new result
                result = s.query(query)
                print("\033[6;0H\033[1;32mResult: \033[0m")
                s.print_result(result)
                if args.o:
                    # write result to file for evaluating D2
                    with open(args.o, 'a') as f:
                        f.write("Query: %s\n" % query)
                        f.write("Result: \n")
                        s.print_result(result, output=f)
                        f.write("\n")
            except IOError:
                print("\n\n\nThis file does not exist")
        elif a == 'e':
            print("\033[1;32mGoodbye!\033[0m")
            break
