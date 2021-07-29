import numpy as np
import os
from Bio.Seq import Seq
from Bio.Alphabet import generic_dna
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqIO.FastaIO import FastaWriter

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
            #print("subseq = "+str(subseq))
            pileOfseqs.append(subseq) #from start to end+1 (not included) there are 58 nucleotides
            start +=1
            end +=1
        #print("getSlidingWindowsSeq - pileOfseqs " + str(pileOfseqs))
        return pileOfseqs

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


def dev_genome(folderIn, endOfTheFiles, NMaxReads=np.Infinity, fileType="fasta", folderOut=None):
    """
    It concatenates a set of genomes/shotgun into a list of lits, each file is a diferent list.

    This function returns a list whose positions are each one of the genomes that were in folder 'folderIn', concatenated. In other words, each list's position is a genome, an unique sequence.
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