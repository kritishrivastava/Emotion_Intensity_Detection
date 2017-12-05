import sys


def create_arff(input_file, output_file):
    """
    Creates an arff dataset
    """

    out = open(output_file, "w" , encoding="utf8")
    header = '@relation ' + input_file + '\n\n@attribute id numeric \n@attribute id1 string \n@attribute tweet string\n@attribute emotion string\n@attribute score numeric\n'
    for i in range(1, 401):
        header += '@attribute c' + str(i) + ' numeric\n'
    header += '\n@data\n'
    out.write(header)

    f = open(input_file, "r", encoding="utf8")
    lines = f.readlines()
    firstline = True
    for line in lines:
        if firstline == True:
            firstline = False
            continue
        # print(line)
        parts = line.split("\t")
        if len(parts) == 405:
            id = parts[0]
            id1 = parts[1]
            tweet = parts[2]
            emotion = parts[3]
            score = parts[4].strip()
            score = score if score != "NONE" else "?"
            out_line = id + ',\"' + id1 + "\"," + '\"' + tweet + '\",' + '\"' + emotion + '\",' + score
            for i in range(5, 405):
                out_line += ',' + parts[i].strip()
            out_line += '\n'
            out.write(out_line)
        elif len(parts) == 1:
            continue
        else:
            print( "Wrong format")
            print(len(parts))
            print(line)
    f.close()
    out.close()


def main(argv):
    input_file = argv[0]
    output_file = argv[1]
    create_arff(input_file, output_file)


if __name__ == "__main__":
    main(sys.argv[1:])