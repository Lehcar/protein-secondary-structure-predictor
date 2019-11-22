file = open("dataset.txt", "r")

sequences = []
structures = []

def parse_data(file, sequences, structures):
    lines = file.readlines()
    new_seq = False
    new_struct = False
    full_seq = ""
    full_struct = ""
    for line in lines:
        if ">" in line:
            # There is a new sequence
            new_seq = True
            new_struct = False
            full_seq = ""
            full_struct = ""
        # if we are in a new sequence and we run into a newline
        # we know that we're getting the secondary structure now
        elif new_seq and line == "\n":
            new_seq = False
            new_struct = True
            sequences.append(full_seq)
        #if we're on a new sequence
        elif new_seq:
            #strip all newline characters and add the sequence to our full sequence
            full_seq += line.strip()
        elif new_struct and line == "\n":
            #we know that we are at the end of the secondary structure
            structures.append(full_struct)
            new_seq = False
            new_struct = False
        elif new_struct:
            full_struct += line.strip()

print(sequences)
print(structures)

