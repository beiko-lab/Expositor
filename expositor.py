from encode_and_predict_genome import EncodeAndPredictGenome
from beiko_lab.DNA_encoders.pc3mer import Pc3mer
from beiko_lab.DNA_encoders.kmers import Kmers
from datetime import datetime
import argparse
import os

parser = argparse.ArgumentParser(description="Apply Expositor to .fna file located in the path -p")
parser.add_argument('-p', type=str, help="Absolute path to the folder where the sequence is", required=True)
parser.add_argument("-gen_size", type=int, help="Number of nucleotides in the sequence inside the .fna file",
                    required=True)
args = parser.parse_args()
path_output = args.p
gen_size = args.gen_size
classes = ["non-promoter", "promoter"]
encoders_to_use = [Kmers(list_of_ks=list(range(1, 6))), Pc3mer(classes=classes)]
model = EncodeAndPredictGenome(
                                encoders_to_use=encoders_to_use,
                                model_folder="",
                                classes=classes,
                                path_db=path_output,
                                path_output=path_output,
                                output_file="output.txt",
                                process_line_by_line = False,
                                min_seq_size_to_search = 58,
                                scaler = "scaler.pkl",
                                path_model = os.path.join(os.getcwd(),"model/"),
                                path_db_suffix = None,
                                path_output_suffix = None
                              )
frag_length = 5000
for j, i in enumerate(range(1, gen_size, frag_length)):
    dt_string = datetime.now().strftime("%Y-%m-%d %H:%M")
    print("\n"+dt_string + " START evaluating chunck **"+str(j)+"** of the genome")
    model.predict((i, i + frag_length + 57))
    print(str(datetime.now().strftime("%Y-%m-%d %H:%M")) + "FINISHED predictions for chunck **" + str(j) + "**\n")
