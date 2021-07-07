from multiprocessing import Pool

def get_tick_sets ():
    return [ [x] for x in range(100) ]

def read_tick_set (set):
    return [0,set]
    
def main():

    p = Pool(processes=15)
    # this call just builds a list with arguments
    calls = get_tick_sets()
    # this should call read_tick_set with the list of argument lists
    # body of read_tick_set is commented out except for return 
    # so dunno whats up here
    print(calls)
    tick_sets = p.starmap_async(read_tick_set, calls)

    tick_sets.wait()

    p.close()
    p.join()

    vals = tick_sets.get()
    print(vals)

if __name__ == '__main__':
    main()