import numpy as np
import pandas as pd
import os
from Bio.Seq import Seq
from Bio.Alphabet import generic_dna
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqIO.FastaIO import FastaWriter

# from Bioinfo.classificationSeq58Nucleotides.reverseComplementOfAString
def reverseComplementOfAString(seq):
    """
    Calculates the reverse complement of a genome
    :param seq: (str) DNA sequence
    :return: The complement of a DNA sequence in the reverse order. For input = AACTG, output is CAGTT
    """

    seq = seq.upper()
    seq = seq.replace("A","B").replace("T","A").replace("B","T")
    seq = seq.replace("C", "B").replace("G", "C").replace("B", "G")

    return seq[::-1]

# from Bioinfo.classificationSeq58Nucleotides.get_genome
def get_genome(folderGenomeForTest):
    """
    :param folderGenomeForTest: (str) Absolute path to DNA's folder
    :return: The complement of a DNA sequence in the reverse order. For input = AACTG, output is CAGTT
    """
    try:
        eColiCompleteGenome = dev_genome(folderGenomeForTest,".fasta")
    except:
        eColiCompleteGenome = dev_genome(folderGenomeForTest, ".fna")
    eColiCompleteGenome = eColiCompleteGenome[0].upper()
    eColiCompleteGenomeReverseComplement=reverseComplementOfAString(str(eColiCompleteGenome)).upper()
    return eColiCompleteGenome,eColiCompleteGenomeReverseComplement

# from Bioinfo.classificationSeq58Nucleotides.getSlidingWindowsSeq
def getSlidingWindowsSeq(seq,windowsLength = 58):
    '''
     Slides a windows of windowsLength nucleotides over the sequence seq
    :param seq: (str) sequence
    :param windowsLength: (int) number of nucleotides in each output sequence
    :return: overlapping sequences of size windowsLength
    '''
    if len(seq) < windowsLength:
        return []
    else:
        pileOfseqs = []
        start = 0
        end = windowsLength-1
        while(end < len(seq)):
            subseq=seq[start:end+1]
            pileOfseqs.append(subseq) #from start to end+1 (not included) there are 58 nucleotides
            start +=1
            end +=1
        return pileOfseqs

# from raw_data.genome_db.GenomesDB.get_sequences_between_intervals
def get_sequence_between_interval(genome_seq, start, end, strand='F'):
    """
    source: genome_db.py
    :param genome_seq:
    :param start:
    :param end:
    :param strand:
    :return:
    """
    assert start != 0, "Remember that start and end are positions and not indexes!"
    assert end <= len(genome_seq), "Remember that start and end are positions and not indexes!"
    assert strand in ["F","R"], "Inform the strand as follow: 'F' or 'R'"
    seq = genome_seq[start-1:end]
    if strand != "F":
        seq = reverse_complement(seq)
    return seq

# from Libraries.raw_data.fasta.stringlist2fasta
def stringlist2fasta(fileName, folderOut, list_of_segments, list_of_ids=None, lines_length=81):
    """
    Source: fasta.py
    :param fileName:
    :param folderOut:
    :param list_of_segments:
    :param list_of_ids:
    :param lines_length:
    :return:
    """
    assert isinstance(list_of_segments, list) or isinstance(list_of_segments, np.ndarray)
    print("Converting strings into fasta format...")
    fOut = open(folderOut+fileName + ".fasta", "w")

    if list_of_ids is not None:
        assert isinstance(list_of_segments, list) or isinstance(list_of_segments, np.ndarray)
        ids = list_of_ids
    else:
        ids = list(range(len(list_of_segments)))

    reg = []
    for i, seg in enumerate(list_of_segments):
        seg = str(seg)
        if len(seg) > 0:
            reg.append(SeqRecord(Seq(seg,generic_dna), id = str(ids[i]), description="Empty"))

    writer = FastaWriter(fOut, wrap=lines_length)
    writer.write_file(reg)
    fOut.close()
    print("[FILE SAVED] Fasta file at {}.".format(folderOut+fileName + ".fasta"))
    return folderOut+fileName + ".fasta"

# from Bioinfo/Compare_seqs.py
def dev_genome(folderIn, endOfTheFiles, NMaxReads=np.Infinity, fileType="fasta", folderOut=None):
    """
    It concatenates a set of genomes/shotgun into a list of lists, each file is a diferent list.

    This function returns a list whose positions are each one of the genomes that were in folder 'folderIn',
    concatenated. In other words, each list's position is a genome, an unique sequence.
    Source: compare_seqs.py
    Parameters
    ----------
    folderIn : str
        Folder's, which has the genome files, path.
    endOfTheFiles : str
        It's a string with the caracters which end the files we want to concatenate.
    NMaxReads : int
        It's the maximum number of times that a register will be read from the file.
        Default = np.Infinity
    fileType : str
        It's the kind of the files we want to concatenate.
        Default = "fasta"
    folderOut : str
        It's the path to a folder where we want to put the result file.
        Default = folderIn

    Returns
    -------
    list
        List with the entire genome of each organism in each position => list[0] =  individual number 0's genome.

    file
        A file named "concatenated_genome.txt" with the concatenated genomes organized by lines.

    """
    # it will save the output file in the same folder of seqs_file unless another folder is specified
    if folderOut is None:
        folderOut = folderIn

    # files_names = ["GCF_000598145.1_ASM59814v1_genomic.fna", "GCF_000598165.1_ASM59816v1_genomic.fna", "GCF_000598205.1_ASM59820v1_genomic.fna"]
    files_names = [f for f in os.listdir(folderIn) if (f.lower().endswith(endOfTheFiles))]

    if files_names == []:
        exit()

    seqs = []
    f2print = open(folderOut + "ConcatenatedContigs.txt", "w")

    for f in files_names:
        concatenated = Seq("", generic_dna)
        i = 0
        # Parse the files f in folderIn as fileTypes
        for s in SeqIO.parse(folderIn + f, fileType):
            if (i > NMaxReads):
                break
            else:
                concatenated += s.seq
                i += 1
        seqs.append(concatenated)
        f2print.write(str(concatenated) + "\n")

    f2print.close()
    return seqs

# from data_flow/feature_vector_from_preds.py
def read_predictions(predictions_folder, columns_to_parse=["pos", "sigma--", "sigma++"],
                     interval_to_considere=None, seq_length=58):
    """
    Reads the predictions from csv files, drop duplicated positions and creates df for forward and reverse strands
    :param predictions_folder:
    :param columns_to_parse:
    :return:
    Two dataFrames, one for forward and another for reverse strands.
    Each df has the structure: ["pos", "sigma--", "sigma++"]
        where:
            pos = position in the DNA
            sigma-- = probability to belong to class sigma--
            sigma++ = probability to belong to class sigma++
    """

    # get list of the names of the csv files in predictions_folders
    predictions_files = [file for file in os.listdir(predictions_folder) if file.endswith(".csv")]

    assert len(predictions_files) > 0, f"No file was found at {predictions_folder}"

    if columns_to_parse == ["pos", "sigma--", "sigma++"] or columns_to_parse == ["index", "pos", "sigma++"]:
        predictionsF = pd.DataFrame(columns=columns_to_parse)
        predictionsR = pd.DataFrame(columns=columns_to_parse)
        # reads each file into a DataFrame
        for file_path in predictions_files:
            print(f"Reading file {predictions_folder + file_path}")
            file = pd.io.parsers.read_csv(predictions_folder + file_path, header=0, \
                                          index_col=0, \
                                          dtype={"pos": int},
                                          low_memory=False, error_bad_lines=True, \
                                          skiprows=0, memory_map=True, float_precision="high", \
                                          engine='c',
                                          names=columns_to_parse,
                                          )

            # keep only the predictions for positions we are interested on.
            if interval_to_considere is not None:
                file = file[file.pos.between(interval_to_considere[0], interval_to_considere[1])]

            # saves the prediction as prediction for the forward or reverse strand
            if "F.csv" in file_path:
                predictionsF = pd.concat([predictionsF, file], ignore_index=True, sort=False)
            else:  # if "R.csv" in file_path:
                predictionsR = pd.concat([predictionsR, file], ignore_index=True, sort=False)

        # if a position doesn't have all its predictions, i.e., one column is nan, it will be discarded
        predictionsF.replace('', np.nan, inplace=True)
        predictionsF.dropna(inplace=True)
        predictionsR.replace('', np.nan, inplace=True)
        predictionsR.dropna(inplace=True)

        # discard rows if the pos is duplicate keeping the first occurrence.
        predictionsF.drop_duplicates(subset=["pos"], keep='first', inplace=True)
        predictionsR.drop_duplicates(subset=["pos"], keep='first', inplace=True)

        # sorting the dataframes based on the positions
        predictionsF.sort_values(["pos"], axis=0, ascending=True, inplace=True, kind='mergesort', na_position='first')
        predictionsR.sort_values(["pos"], axis=0, ascending=True, inplace=True, kind='mergesort', na_position='first')


        print(f"Found predictions for {len(predictionsF)} positions on forward\n"+ \
              f"Found predictions for {len(predictionsR)} positions on reverse\n")

    else:
        print("This method read_predictions is not implemented to parse your files columns.\n")
        exit()

    if columns_to_parse == ["index", "pos", "sigma++"]:
        predictionsF["sigma--"] = 1 - predictionsF["sigma++"]
        predictionsR["sigma--"] = 1 - predictionsR["sigma++"]

        predictionsF = predictionsF.loc[:, ["pos", "sigma--", "sigma++"]]
        predictionsR = predictionsR.loc[:, ["pos", "sigma--", "sigma++"]]

    return predictionsF, predictionsR

# from data_manipulation/sequences_and_intervals.py
def moving_average(seqlen: int, pos_start: int, pos_end: int, preds):
    """
    :param seqlen (int): number of nucleotides to be consider for the average including the current position.
    mvg_avg(x) = mean(preds["pos"].between(x - (seqlen-1), x, inclusive=True))
    Ex. when calculating the average position for position 58 when seqlen = 58,
    mvg_avg(58) = mean(preds["pos"].between(1, 58, inclusive=True))
    :param pos_start (int): position where the moving average should start
    :param pos_end (int): position where the moving average should end
    :param preds (pd.DataFrame): DataFrame with column 'pos' where the positions are and
    column 'sigma++' where the probabilities of each position to be in a promoter are
    :return:
    """
    # x axis to plot the moving average
    min_start_pos = max(seqlen, pos_start) # finds the minimum position where it makes since to start from

    # average(current positions back to seqlen-1 positions before it)
    # one way to do this with for
    # yy = [np.average(preds[preds["pos"].between(x - (seqlen - 1), x, inclusive=True)][["sigma++"]].values) for x in
    #       xx]
    # another way to do it with less loops:
    almost_y = preds[preds["pos"].between(min_start_pos-(seqlen-1), pos_end, inclusive=True)][["pos","sigma++"]]
    y = almost_y[["sigma++"]].values
    xx = almost_y[["pos"]].values.flatten()

    matrix = np.array(y).reshape([len(y), 1])
    for i in range(1, seqlen):
        y_ = np.zeros([len(y), 1])
        y_[i:] = y[:-i]
        matrix = np.concatenate([matrix,y_], axis=1)

    yy = np.sum(matrix, axis=1)/seqlen
    yy = yy[seqlen-1:]
    xx = xx[seqlen - 1:]

    # we need the add seqlen-1 zeros at the begining of yy and xx
    if min_start_pos == seqlen:
        xx = np.concatenate([list(range(1,seqlen)), xx])
        yy = np.concatenate([np.zeros(seqlen-1), yy])

    if xx[0] > min_start_pos:
        add_to_the_begining = xx[0] - min_start_pos
        xx = np.concatenate([list(range(min_start_pos,xx[0])), xx])
        yy = np.concatenate([np.zeros(add_to_the_begining), yy])

    assert len(xx) == len(yy)
    return xx, yy

# from data_manipulation/sequences_and_intervals.py
def get_intervals_from_array_of_ones(y_pred, array_initial_position):
    """
    Received an array of zeros and ones and it returns the interval where the ones are
    :param y_pred: array of zeros and ones
    :param array_initial_position: it says where y_pred predictions start at the genome
    :return: np array with np arrays of size 2: [start, end] of each interval.
    Ex. [[45,78],[88, 458]]
    """
    indexes = np.where(np.array(y_pred) == 1)[0]
    prom_ranges = []
    i = 0
    while i < len(indexes):
        start_new_interval = indexes[i] + array_initial_position
        while (i+1 < len(indexes)) and (indexes[i + 1] == indexes[i] + 1):
            i += 1
        prom_ranges.append(np.array([start_new_interval, indexes[i] + array_initial_position]))
        i+=1

    return np.array(prom_ranges)