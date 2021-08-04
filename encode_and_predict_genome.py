import tensorflow as tf
import os
import numpy as np
from utils import get_genome, getSlidingWindowsSeq, get_sequence_between_interval, stringlist2fasta
import pandas as pd
from datetime import datetime
import time
import keras
from glob import glob
from Bio.Seq import reverse_complement
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""
Level | Level for Humans | Level Description                  
 -------|------------------|------------------------------------ 
  0     | DEBUG            | [Default] Print all messages       
  1     | INFO             | Filter out INFO messages           
  2     | WARNING          | Filter out INFO & WARNING messages 
  3     | ERROR            | Filter out all messages    
  """


class EncodeAndPredictGenome:
    """
        Breaks E.coli's genome into fragment, encodes with the list of encoders objects and predicts with model inside
        model_folder

        Parameters
        ----------
        encoders_to_use : List
            Encoders objects inheriting from Libraries.encoders.base_for_encoders.BaseForEncoder.
            They will be used with a call to the method:
             encode_fasta_file(fasta file path, output path, add_class_to_entries=False).
        model_folder : String
            Folder name from right after "PromoterOutputs/building_models/" to where the model .h5 or .pkl is.
        classes : List of strings
            List of the classes following the same order as used to train the model in model_folder
        path_db: String
            Path to the database, i.e., path to where the PromoterDatabase folder is
        path_output: String
            Path to the outputs, i.e., path to where the PromoterOutputs folder is
        process_line_by_line : Boolean, default True
            If True, each sequences with min_seq_size_to_search nucleotides is process line by line.
            If False, all the sequences with min_seq_size_to_search nucleotides found inside >>>predict(pos2search)
            is processed together and saved together.
        min_seq_size_to_search : Integer, default 58
            Sliding window length
        scaler : String
            Name of the file containing the scaler used to standardize the data for training
        path_model : String
            Path to the model
        path_db_suffix : String
            String to be used if you want to add to path_db more specification to where the genome is
        path_output_suffix : String
            String to be used if you want to add to path_output more specification to where to save the results

        See Also
        --------
        EncodeAndPredictGenome.predict : Initiate the breaking, encoding and predicting process

        Examples
        --------
        Creating an encode_and_predict_genome object
        >>>obj = EncodeAndPredictGenome([kmers_object_instantiated, pseknc_object_instantiated],\
                    "./model1/", ["Sigma--", "Sigma++"], process_line_by_line=True,\
                    min_seq_size_to_search=58))
        1. The sliding window to be used is 58 nucleotides long
        2. The model in "./model1/" was trained using [1,0] as Sigma-- and [0,1] as Sigma++
        3. Each sequence of 58 nucleotides will be saved into fasta, encoded into kmers and saved,
        encoded into pseknc and saved, and predicted and saved.

        Calling predict to E. coli's genome interval.
        >>>obj.predict((1,100))
        It will generated 100-57 sequences of 58 nucleotides and create:
        fasta/
            one fasta file for each sequence - forward
            one fasta file for each sequence - reverse
        encoder_1/
            one file for each sequence encoded with encoder_1 - forward
            one file for each sequence encoded with encoder_1 - reverse
        encoder_2/
            one file for each sequence encoded with encoder_2 - forward
            one file for each sequence encoded with encoder_2 - reverse
        predictions/
            one file with prediction for each sequence - forward
            one file with prediction for each sequence - reverse

        Creating object to save execution time by processing bigger intervals each time
        >>>path_output = ...path where PromoterOutputs ...
        # later, path_output will be updated to path_output + path_output_suffix
        # and the model will be uploaded from path_output + model_folder
        >>>model_folder = "17v_03_05_v009_2ndrun/best_model/"
        >>>classes = ["Sigma--", "Sigma++"]
        >>>encoders_to_use = [Kmers(list_of_ks=list(range(1, 6))), Pc3mer(classes=classes)]
        >>>model = EncodeAndPredictGenome(encoders_to_use, model_folder, classes, process_line_by_line=False)
        >>>start = 27178
        >>>for i in range(26):
        >>> model.predict((start + i * 5000 , start + (i + 1) * 5000))
        All the sequences found inside the interval from start+i*5000 to start+(i+1)*5000 will be processed
        together
        """

    def __init__(self, encoders_to_use: list, model_folder: str, classes: list, path_db: str, path_output: str,
                 process_line_by_line=True, min_seq_size_to_search=58, scaler=None, path_model=None,
                 path_db_suffix="PromoterDatabase/E.coliK12ForTest/",
                 path_output_suffix="PromoterOutputs/testing_on_genome/"):

        if path_output_suffix is not None:
            path_output_testing_on_genome = path_output + path_output_suffix
        else:
            path_output_testing_on_genome = path_output
        if path_model is None:
            path_model = path_output + "PromoterOutputs/building_models/"

        self.encoders_to_use = encoders_to_use
        self.configs = \
            {
                "path_db": path_db,
                "path_model": path_model,
                "path_output_testing_on_genome": path_output_testing_on_genome,
                "min_seq_size_to_search": min_seq_size_to_search,
                "pos2search": None,
                "windowsOffSet4Preds": 1,
                "classes": classes,
            }

        '''
        Defining constraints
        '''
        if path_output_suffix is not None:
            folderGenomeForTest = self.configs["path_db"] + path_db_suffix
        else:
            folderGenomeForTest = self.configs["path_db"]

        self.eColiCompleteGenome, self.eColiCompleteGenomeReverseComplement = get_genome(
            folderGenomeForTest)
        self.genome_size = len(self.eColiCompleteGenome)

        """
        Model
        """
        print(f'Loading model ...')
        model_path = self.configs["path_model"] + model_folder
        # load model
        with tf.device('/gpu:0'):
            try:
                self.clf = tf.contrib.predictor.from_saved_model(model_path)
            except:
                keras.backend.clear_session()
                self.clf = keras.models.load_model(glob(model_path + "*.h5")[0])

        if scaler:
            fid=open(self.configs["path_model"] + scaler, "rb")
            self.scaler = pickle.load(fid)
        else:
            self.scaler=None
        print(f'Model {model_path} successfully loaded.')


        """
        Outputs
        """
        self.folderForResults = self.configs["path_output_testing_on_genome"] + model_folder
        os.makedirs(self.folderForResults, exist_ok=True)

        self.process_line_by_line = process_line_by_line


    def saveDFPredictionsInHD(self, preds, predsR, folderForResults, pos2search=None):
        folderForResults = folderForResults+"predictions/"
        os.makedirs(folderForResults, exist_ok=True)

        if pos2search is not None:
            fileName = folderForResults +f"pos_{pos2search[0]}"+\
                       f"_{pos2search[1]-self.configs['min_seq_size_to_search']}"
        else:
            fileName = folderForResults+datetime.now().strftime("%Y-%m-%d_%H-%M")

        preds.to_csv(fileName+"_F.csv",header = True)
        predsR.to_csv(fileName+"_R.csv",header = True)

        print("Predictions saved in {}".format(fileName))

    def predict(self, pos2search: tuple):
        print(f"Predict called for the interval {pos2search}")
        positions, predsA, predsB = self.predictingForBothStrandDirection(pos2search)

        positions = positions.reshape([len(positions),1])
        columns = ["pos"]
        columns.extend(self.configs["classes"])
        predsF = pd.DataFrame(data = np.concatenate([positions, predsA], axis=1), columns=columns)
        predsR = pd.DataFrame(data = np.concatenate([positions, predsB], axis=1), columns=columns)
        #saving predictions
        self.saveDFPredictionsInHD(predsF, predsR, self.folderForResults, pos2search=pos2search)


    def predictingForBothStrandDirection(self, pos2search: tuple):
        '''
        :param pos2search: Tuple: (1126,1127) - 1127 inclusive
        :return: The return only returns the predictions for the last intergene in the intergenesToSearch
        '''
        # safely stablishes the intervals of nucleotides (sequences) that should be search
        minSeq2search = self.configs["min_seq_size_to_search"]
        if (pos2search == "all"):
            start, end = (0, self.genome_size)
        else:
            start, end = pos2search
            if (end - start < minSeq2search):
                print("The sequence length must be bigger than" + str(minSeq2search - 1) + "nucleotides")
                return

        preds = []
        predsR = []

        # Grabbing the intervals to then extract and handle the sequences
        positions_we_are_predicting_for = np.array(range(start, end - minSeq2search + 1, self.configs["windowsOffSet4Preds"]))
        intervals = [(i, i + minSeq2search - 1) for i in positions_we_are_predicting_for]

        if not self.process_line_by_line:
            preds = self.predicting_chuncks(pos2search, intervals, strand='F')
            predsR = self.predicting_chuncks(pos2search, intervals, strand='R')
        else:
            for oneIntergeneEachTime in intervals:
                print(f"Working on fragment {oneIntergeneEachTime}")
                # # Classifying the forward strand
                preds.append(np.array(self.predict_sub(oneIntergeneEachTime, strand='F'))[0])

                # # Classifying the reverse strand
                predsR.append(np.array(self.predict_sub(oneIntergeneEachTime, strand='R'))[0])

        preds = np.array(preds)
        predsR = np.array(predsR)

        assert len(preds) == len(positions_we_are_predicting_for)
        assert len(predsR) == len(positions_we_are_predicting_for)

        return positions_we_are_predicting_for, preds, predsR

    def predicting_chuncks(self,start_end, intervals, strand='F'):
        print(f"Interval {start_end} - {strand}-strand.")
        seqs=[]
        for start, end in intervals:
            s= get_sequence_between_interval(self.eColiCompleteGenome, start, end)
            if strand == 'R':
                s = reverse_complement(s)
            seqs.append(s)

        positions_to_name_the_files = "position=" + \
                        str(start_end[0]) + "_to_" + \
                        str(start_end[0] + len(seqs) - 1) + "_" + str(strand)

        # saving sequences into a fasta file
        os.makedirs(self.folderForResults + "fasta/", exist_ok=True)
        fastaFileName = self.folderForResults + "fasta/" + positions_to_name_the_files
        stringlist2fasta(fastaFileName, '', seqs,
                         list_of_ids=list(range(start_end[0], start_end[0] + len(seqs))), lines_length=58)

        return self.encode_and_predict_sequences(fastaFileName, positions_to_name_the_files)

    def predict_sub(self, start_end, strand='F'):
        print(f"{strand}-strand.")
        # Grabbing the sequence
        seq = get_sequence_between_interval(self.eColiCompleteGenome,
                                                      start_end[0], start_end[1])
        if strand == 'R':
            seq = reverse_complement(seq)

        # cutting sequence into fragments of 58 nucleotides and saving the fasta file
        pileOfSeqs = getSlidingWindowsSeq(seq, windowsLength=self.configs["min_seq_size_to_search"])
        if pileOfSeqs == []:
            return
        else:
            os.makedirs(self.folderForResults+"fasta/", exist_ok=True)
            positions_to_name_the_files = "position=" + str(start_end[0]) + str(strand)
            fastaFileName = self.folderForResults+"fasta/" + positions_to_name_the_files
            stringlist2fasta(fastaFileName, '', pileOfSeqs,
                             list_of_ids=list(range(start_end[0], start_end[0] + len(pileOfSeqs))), lines_length = 58)

        return self.encode_and_predict_sequences(fastaFileName, positions_to_name_the_files)

    def encode_and_predict_sequences(self, fastaFileName, positions_to_name_the_files):
        #Try to read the encoding but creates it if not found
        encoded = []
        for i, encoder in enumerate(self.encoders_to_use):
            num_encoders=i+1
            output_path = self.folderForResults+"encoder_{}/".format(num_encoders)
            os.makedirs(output_path, exist_ok=True)
            encoded_file_path = output_path + positions_to_name_the_files
            if not os.path.isfile(encoded_file_path):
                encoder.encode_fasta_file(fastaFileName + '.fasta', encoded_file_path, add_class_to_entries=False)
            if len(encoded)==0:
                encoded = np.array(pd.read_csv(encoded_file_path, header=None, index_col=None))
            else:
                encoded = np.concatenate([encoded,
                                          pd.read_csv(encoded_file_path, header=None, index_col=None).values], axis=1)

        if self.scaler:
            encoded = self.scaler.transform(encoded)
        else:
            print("[LOCATION] encode_and_predict_genome.EncodeAndPredictGenome.encode_and_predict_sequences")
            print("No scaling is been applied to the data")

        # predicting for feature vector
        if (len(encoded) > 0):
            with tf.device('/gpu:0'):
                try:
                    predictions = self.clf({"input": encoded})
                except:
                    predictions = self.clf.predict(encoded, batch_size=len(encoded), verbose=1)
            return predictions

