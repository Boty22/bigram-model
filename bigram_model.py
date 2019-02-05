# python homework.py corpus.txt "Facebook announced plans to built a new datacenter in 2018 ." "Facebook is an American social networking service company in California ."

import nltk
import sys
from math import log, exp

def generate_model(raw):
    #converting the words to their lower cases
    #raw = raw.lower()
    model={}
    #Extracting the sentences from the corpus
    sentences = nltk.sent_tokenize(raw) 
    
    #Saving the number of senteces in the corpus
    n_sentences = len(sentences)      
    model['n_sentences'] = n_sentences
    
    #Extracting the first word of each sentence and the tokenizing raw
    first_words_list = []
    tokens = []
    for sent in sentences:
        aux = nltk.word_tokenize(sent.lower())
        first_words_list.append(aux[0])
        tokens = tokens + aux

    #Saving    
    model['fd_tokens'] = nltk.FreqDist(tokens)
    model['V'] = len(model['fd_tokens'].keys()) 
    
    #Saving the frequencie distributions for Non-smooting model
    model['fd_first_words'] = nltk.FreqDist(first_words_list)
    model['fd_bigrams'] = nltk.FreqDist(list(nltk.bigrams(tokens)))
   
    return model

def get_data_sentence(sentence):
    tokens_sentence = [w.lower() for w in nltk.word_tokenize(sentence)]
    bigrams_sentence = list(nltk.bigrams(tokens_sentence))
    fd_tokens = nltk.FreqDist(tokens_sentence)
    sent={}
    sent['tokens'] = tokens_sentence
    # Note: sent['tokens'][0] is the first word of the sentence
    sent['bigrams'] = bigrams_sentence
    sent['fd_tokens'] = fd_tokens
    return sent

def generate_bigram_tables(sent_data, model):
    #the keys of the fd of the sent_data will be stores in words
    words = sent_data['fd_tokens'].keys()

    '''
    #Test block
    print('\n1) Table of counts for individual words:\n')
    
    print('|{0:^10}|{1:^10}|'.format('WORD','COUNTS'))
    for word in words:
        print('|{0:<10}|{1:>10}|'.format(word, model['fd_tokens'][word]))
    '''        
    #  A1) Table of counts for bigram model without smoothing:
    #-----------------------------------------------------------

    print('\nA1) Table of counts for bigram model without smoothing:\n')
    #header
    print('{0:^11}|'.format('BIGRAMS'), end='')
    for word in words:
        print('{0:^10}|'.format(word), end='')
    print('')
    #body rows
    for w1 in words:
        #row word
        print('|{0:<10}|'.format(w1), end='')
        #counts for each bigram
        for w2 in words:
            print('{0:>10}|'.format(model['fd_bigrams'][(w1,w2)]), end= '')
        print('')
        
    #  A2) Table of counts for bigram model with smoothing:
    #-----------------------------------------------------------

    print('\nA2) Table of counts for bigram model with smoothing:\n')
    #header
    print('{0:^11}|'.format('BIGRAMS'), end='')
    for word in words:
        print('{0:^10}|'.format(word), end='')
    print('')
    #body rows
    for w1 in words:
        #row word
        print('|{0:<10}|'.format(w1), end='')
        #counts for each bigram
        for w2 in words:
            print('{0:>10}|'.format(model['fd_bigrams'][(w1,w2)] + 1), end= '')
        print('')
        

        
def generate_probs(sent_data, model):
    #Number of sentences in the corus of the model
    n_sent_corpus = model['n_sentences']
    V = model['V']

    #Words in the sentence
    words = sent_data['fd_tokens'].keys()
    #first word in the sentence
    first_word = sent_data['tokens'][0]
    #bigrams in the sentence 
    s_bigrams = sent_data['bigrams']

    
    #  B1) Probabilities for the bigrmas without smoothing:
    #---------------------------------------------------
    print('\nB1) Table with probabilities bigrams without smoothing:\n')
    prob_bigrams_no_smooth = {}
    
    #header
    print('{0:^11}|'.format('BIGRAMS'), end='')
    for word in words:
        print('{0:^10}|'.format(word), end='')
    print('')
   
    #body rows
    for w1 in words:
        #row word
        print('|{0:<10}|'.format(w1), end='')
        #counts for each bigram
        for w2 in words:
            p = model['fd_bigrams'][(w1,w2)] / model['fd_tokens'][w1]
            if(p>0):
                print('{0:<10.6f}|'.format(p), end= '')
            else:
                print('{0:<10.1f}|'.format(p), end= '')
            prob_bigrams_no_smooth[(w1,w2)] = p
        print('')
        
    
    # B2) Probabilitis for the bigrmas with smoothing:
    #----------------------------------------------
    print('\nB2) Table with probabilities bigrams with smoothing:\n')
    prob_bigrams_with_smooth = {}
    
    #header
    print('{0:^11}|'.format('BIGRAMS'), end='')
    for word in words:
        print('{0:^10}|'.format(word), end='')
    print('')
   
    #body rows
    for w1 in words:
        #row word
        print('|{0:<10}|'.format(w1), end='')
        #Probs for each bigram
        for w2 in words:
            p = (model['fd_bigrams'][(w1,w2)] + 1)/ (model['fd_tokens'][w1]+ V)
            if(p>0):
                print('{0:<10.6f}|'.format(p), end= '')
            else:
                print('{0:<10.1f}|'.format(p), end= '')
            prob_bigrams_with_smooth[(w1,w2)] = p
        print('')        
        
    
    # C1) Getting probabilities without smoothing:
    #----------------------------------------
    p_any_sent = 1/n_sent_corpus
    #Probability of being the first word
    p_fs = model['fd_first_words'][first_word] / n_sent_corpus 

    #print(first_word, p_fs)

    prob_sentence = p_any_sent * p_fs
    for b in s_bigrams:
        #print(b,prob_bigrams_no_smooth[b])
        prob_sentence *= prob_bigrams_no_smooth[b]
    
    print('\nC1) The probability of the sentence is without smoothing:', prob_sentence)
    
    

    #  C2) Getting probabilities with smoothing:
    #----------------------------------------
    log_p_any_sent = log(1/n_sent_corpus)
    #Probability of being the first word
    log_p_fs = log( (model['fd_first_words'][first_word]+1) / (n_sent_corpus+V) )

    #print(first_word, exp(log_p_fs))

    log_prob_sentence = log_p_any_sent + log_p_fs
    for b in s_bigrams:
        #print(b,prob_bigrams_with_smooth[b])
        log_prob_sentence += log(prob_bigrams_with_smooth[b])
    
    print('\nC2) The probability of the sentence is with smoothing :', exp(log_prob_sentence))


def main():    
	#print('Number of arguments:', len(sys.argv), 'arguments.')
	#print('Argument List:', str(sys.argv))
	input_file_name = sys.argv[1]
	sentence1 = sys.argv[2]
	sentence2 = sys.argv[3]

	"""" --------------------------------------------------------------------------------- 
	input_file_name = "corpus.txt"
	sentence1 = "Facebook announced plans to built a new datacenter in 2018 ."
	sentence2 = "Facebook is an American social networking service company in California ."

	================================================================================== """


	#print('Input File: ',input_file_name)



	raw = ''
	with open(input_file_name, encoding ="utf8" ) as f:
	    line = f.read()
	    line = line.rstrip('\n')
	    raw = raw+line

	#print ('Reading file ....[done]')
	m = generate_model(raw)

	print('\n{0:<85}'.format('='*75))
	print('Sentence 1: ', sentence1)
	print('{0:<85}'.format('='*75))   
	s1 = get_data_sentence(sentence1)
	generate_bigram_tables(s1,m)
	generate_probs(s1,m)

	print('\n{0:<85}'.format('='*88)) 
	print('Sentence 2: ', sentence2)
	print('{0:<85}'.format('='*88))   
	s2 = get_data_sentence(sentence2)
	generate_bigram_tables(s2,m)
	generate_probs(s2,m)


main()