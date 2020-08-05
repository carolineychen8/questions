import nltk
import sys
import string
import numpy as np
import os
import functools

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """

    files = dict()
    filesnew = os.listdir(directory)
    for file in filesnew:
        with open(os.path.join(directory, file)) as f:
            data = f.read()
            files[file] = data
    return files

def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    temp = nltk.word_tokenize(document)
    tokens = [t.lower() for t in temp]
    for token in tokens.copy():
        if token in string.punctuation:
            tokens.remove(token)
        if token in nltk.corpus.stopwords.words("english"):
            tokens.remove(token)
    return tokens

def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.

    idf = ln(num docs / num docs w word)
    """
    #setup
    idfs = dict()
    for doc in documents:
        for word in documents[doc]:
            idfs[word] = 0
            
    numdocs = len(documents.keys())
    for word in idfs.keys():
        numappear = 0
        for doc in documents:
            if word in documents[doc]:
                numappear+=1
        idfs[word] += np.log(numdocs/numappear)

    return idfs

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    stats = []
    #go thru each file and compute tfidf
    for file in files:
        tfidf = 0
        #for each word in file and query, add tfidf value
        for word in query:
            if word in idfs:
                num=idfs[word]
                num*=tf(files[file], word)
                tfidf+=num
        stats.append((file, tfidf))

    #sort by tfidf (highest first)
    stats = sorted(stats, key=lambda stat: -stat[1])

    names = []
    for name, num in stats:
        names.append(name)
        if len(names)==n:
            break
    return names

def tf(file, word):
    #computes term frequency
    count = 0
    for w in file:
        if word==w:
            count+=1
    return count
 
def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    #assembling
    top = []
    for sentence in sentences:
        mwm = matchwordmeasure(query, sentences[sentence], idfs)
        qtd = querytermdensity(query, sentences[sentence])
        top.append((sentence, mwm, qtd))

    #sort by mwm and qtd (highest first)
    top = sorted(top, key=functools.cmp_to_key(compare))
        
    #take top n
    realtop = []
    for s in top:
        realtop.append(s[0])
        if len(realtop) == n:
            break
    return realtop

    
def compare(sentence1, sentence2):
    mwm1 = sentence1[1]
    mwm2 = sentence2[1]
    qtd1 = sentence1[2]
    qtd2 = sentence2[2]
    if mwm1!=mwm2:
        return mwm2-mwm1
    return qtd2-qtd1
    
def matchwordmeasure(query, sentence, idfs):
    """
    sum of IDF values for any word in the query
    that also appears in the sentence
    """
    s = 0
    for word in query:
        if word in sentence:
            s+=idfs[word]
    return s

def querytermdensity(query, sentence):
    """
    proportion of words in the sentence
    that are also words in the query
    """
    totalnum = len(sentence)
    numwords = 0
    for word in sentence:
        if word in query:
            numwords+=1
    return numwords/totalnum

if __name__ == "__main__":
    main()
