import PyPDF2
import numpy as np
import torch
import nltk
from nltk.tokenize import sent_tokenize
from pytorch_pretrained_bert import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from youtube_transcript_api import YouTubeTranscriptApi as yta
from blingfire import text_to_sentences
from deepmultilingualpunctuation import PunctuationModel
import speech_recognition as sr
import pytesseract
import imageio.v2 as iio
pytesseract.pytesseract.tesseract_cmd = r'D:\2023\Final year project\tesseract.exe'

def bertSent_embeding(sentences):
    """
    Input a list of sentence tokens
    Output a list of vectors, each vector is a sentence representation
    """

    print("Adding [CLS] & [SEP]...")
    ## Add sentence head and tail as BERT requested
    marked_sent = ["[CLS] " + item + " [SEP]" for item in sentences]

    print("BERT Token embedding...")
    ## Bert tokenizization
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    tokenized_sent = [tokenizer.tokenize(item) for item in marked_sent]

    print("Transformer positional embedding...")
    ## Giving IDs t0 word tokens embedding using BERT vocabulary
    indexed_tokens = [tokenizer.convert_tokens_to_ids(item) for item in tokenized_sent]
    ## Converting into Tensors
    tokens_tensor = [torch.tensor([item]) for item in indexed_tokens]


    ## Adding segment id as BERT requested
    segments_ids = [[1] * len(item) for ind, item in enumerate(tokenized_sent)]
    segments_tensors = [torch.tensor([item]) for item in segments_ids]


    print("Sending tokens into BERT pre-trained model...")
    ## load BERT base model and set to evaluation mode
    bert_model = BertModel.from_pretrained('bert-large-uncased')
    bert_model.eval()

    ## We get the output from the last encoder layer of BERT
    assert len(tokens_tensor) == len(segments_tensors)
    encoded_layers_list = []
    for i in range(len(tokens_tensor)-1):
        with torch.no_grad():    #Disabling gradient calculation, reducing memory consumption
            encoded_layers, _ = bert_model(tokens_tensor[i], segments_tensors[i])
        encoded_layers_list.append(encoded_layers)

    print("Extracting the last layer vector from the BERT model...")
    ## Use only the last layer vetcor
    token_vecs_list = [layers[23][0] for layers in encoded_layers_list]

    ## Conveting individual word vectors of a sentence into a single vector using mean
    sentence_embedding_list = [torch.mean(vec, dim=0).numpy() for vec in token_vecs_list]

    return sentence_embedding_list


def kmeans_sumIndex(sentence_embedding_list):
    """
    Input a list of embeded sentence vectors
    Output an list of indices of sentence in the paragraph, represent the clustering of key sentence
    """
    n_clusters = np.ceil(len(sentence_embedding_list) ** 0.6)
    kmeans = KMeans(n_clusters=int(n_clusters))
    kmeans = kmeans.fit(sentence_embedding_list)

    sum_index, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, sentence_embedding_list, metric='cosine')

    sum_index = sorted(sum_index)

    return sum_index


def bertSummarize(sentences):
    """
    Input a paragraph as string
    Output the summary including a few key sentences using BERT sentence embedding and clustering
    """

    print("=========================================================")

    sentence_embedding_list = bertSent_embeding(sentences)

    print("Sentence embedded...passing it to Kmeans")

    sum_index = kmeans_sumIndex(sentence_embedding_list)

    summary = '9999'.join([sentences[ind] for ind in sum_index])
    summary = summary.replace('9999','\n')

    return summary
