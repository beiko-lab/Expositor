# orig pp2-138
from encode_and_predict_genome import EncodeAndPredictGenome
from beiko_lab.DNA_encoders.pc3mer import Pc3mer
from beiko_lab.DNA_encoders.kmers import Kmers
from datetime import datetime
import argparse
import os
from utils import *

# Parsing arguments
parser = argparse.ArgumentParser(description="Apply Expositor to .fna file located in the path -p")
parser.add_argument('-p', type=str, help="Absolute path to the folder where the sequence is", required=True)
parser.add_argument("-l", type=int, help="Number of nucleotides in the sequence inside the .fna file",
                    required=True)
parser.add_argument("-m", type=int, help="Minimal threshold limiting the number of nucleotides on a predicted promoter.",
                    default=20)
parser.add_argument("-s", type=bool, help="Whether to store intermediate results when optional or not.",
                    default=False)
args = parser.parse_args()
path_output = args.p
gen_size = args.l
minimal_promoter_length = args.m
store_intermediate_results = args.s

# Defining variables
classes = ["non-promoter", "promoter"]
encoders_to_use = [Kmers(list_of_ks=list(range(1, 6))), Pc3mer(classes=classes)]

# Getting probabilities from the model
model = EncodeAndPredictGenome(encoders_to_use=encoders_to_use, model_folder="", classes=classes, path_db=path_output,
                               path_output=path_output, process_line_by_line=False, min_seq_size_to_search=58,
                               scaler="scaler.pkl", path_model=os.path.join(os.getcwd(), "model/"), path_db_suffix=None,
                               path_output_suffix=None)
frag_length = 5000
for j, i in enumerate(range(1, gen_size, frag_length)):
    dt_string = datetime.now().strftime("%Y-%m-%d %H:%M")
    print("\n"+dt_string + " START evaluating chunck **"+str(j)+"** of the genome")
    end = i + frag_length + 57
    if end > gen_size:
        end=gen_size
    model.predict((i, end))
    print(str(datetime.now().strftime("%Y-%m-%d %H:%M")) + "FINISHED predictions for chunck **" + str(j) + "**\n")




# Merging predictions
"""
Reads the predictions made for L1 on E. coli's genome and store on one file
"""
print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M')} Combining predictions")
predictions_path = path_output+"/predictions/"
# format of predictions ["pos", "sigma--", "sigma++"]
predictionsF, predictionsR = read_predictions(predictions_path, columns_to_parse=["pos", "sigma--", "sigma++"])

# adding strand information to the database
predictionsF['strand'] = ["F"]*len(predictionsF)
predictionsR['strand'] = ["R"]*len(predictionsR)
probabilities = pd.concat([predictionsF, predictionsR], ignore_index=True, axis=0)
probabilities.sort_values(by='pos',inplace=True)

if store_intermediate_results:
    probabilities.to_csv(path_output+"probabilities.csv", header=True)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M')} Probabilities saved at {path_output}probabilities.csv\n")




# Applying moving average.
print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M')} Calculating moving average")
"""
Calculates the  moving average for each pos_x of each strand.
    mvg_avg[pos_x] = sum(probs[pos_x-57:pos_x])/58 
To be in a promoter, a position must have the sum of both moving averages above 100%
"""
df = probabilities

# moving average
mvg_avgs = []
for strand in df.groupby("strand"):
    _, y = np.array(moving_average(58, 1, gen_size, strand[1]))
    mvg_avgs.append(y)

if store_intermediate_results:
    with open(path_output+'/moving_avg.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(mvg_avgs[0])
        writer.writerow(mvg_avgs[1])
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M')} Moving average saved at {path_output}moving_avg.csv")


# converting moving average to array of ones
array_of_ones = [1 if (f+r) >= 1 else 0 for f, r in
                 zip(mvg_avgs[0], mvg_avgs[1])]
del mvg_avgs

if store_intermediate_results:
    with open(path_output+'/array_of_ones.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(array_of_ones)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M')} Genome-size array of zeros and ones. "+
          f"The promoters are represented by the number one. It is saved at {path_output}array_of_ones.csv")


# getting prediction intervals
preds_intervals = get_intervals_from_array_of_ones(array_of_ones, 1)
del array_of_ones

if store_intermediate_results:
    pd.DataFrame(preds_intervals, columns=["start", "end"]). \
        to_csv(path_output+'/promoter_positions_before_filter.csv',
               header=True, index=False)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M')} All promoter predictions' positions saved at {path_output}promoter_positions_before_filter.csv")


# getting final promoters
final_preds = [i for i in preds_intervals if i[1] - i[0] > minimal_promoter_length]
pd.DataFrame(preds_intervals, columns=["start", "end"]). \
    to_csv(path_output+'/final_promoter_positions.csv',
           header=True, index=False)
print(f"\n# {datetime.now().strftime('%Y-%m-%d %H:%M')} **Final predicted promoters** Promoters equal or longer than"+
      f" {minimal_promoter_length} nucleotides saved at {path_output}final_promoter_positions.csv")