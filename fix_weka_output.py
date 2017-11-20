# fix_weka_output.py
# Running example: python fix_weka_output.py data/wekapredictions.csv data/wekapredictionsfixed.csv
import sys

def fix_output(input_file,output_file):
    """
    Fixes the output of a Weka classifier
    """
    f=open(input_file, "r")
    lines=f.readlines()
    f.close()
    
    out=open(output_file,"w") 
    for line in lines[1:len(lines)]:
        parts=line.split("\t")
        if len(parts)==7:
            out.write(parts[4]+'\t'+parts[5][1:len(parts[5])-1]+'\t'+parts[6].strip()+'\t'+parts[2]+'\n')
    out.close()
            
    
   
    
def main(argv):
    input_file=argv[0]
    output_file=argv[1]
    fix_output(input_file,output_file)
   
        
if __name__ == "__main__":
    main(sys.argv[1:])