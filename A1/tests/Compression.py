#write a code to count the nubmer of intergers in a file

#input is the .dat file with integers spaced separated
def count_integers(filename):
    file = open(filename, 'r')
    count = 0
    for line in file:
        count += len(line.split())
    return count

t1 = count_integers('D_medium.dat')
t2 = count_integers('D_medium_out.dat')
print(t1)
print(t2)
print(t2/t1)