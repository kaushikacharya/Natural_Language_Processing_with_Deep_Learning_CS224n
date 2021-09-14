**1g - Sentence Masks**

Attention scores are computed by taking dot product of decoder hidden vector with projected encoder hidden vector. Vector products are computed over batch of sentences. As these sentences may have variable word length, padded tokens are added to shorter sentences.
 
Masks are used to ensure zero contribution to attention scores from these padded tokens.
e_t values corresponding to padded tokens are assigned -Inf. 
As exp(-inf)=0, hence alpha_t (=softmax(e_t)) gets zero for these padded tokens.
 
**1j - Advantages/Disadvantages of attention mechanisms**

Dot product attention:

    Advantage: Whereas other two mechanisms need a weight matrix to trained, dot product attention is simpler as it doesn't needs any weight matrix.
 
Multiplicative attention:

    Advantage: With respect to additive attention, multiplicative attention is faster and more space efficient as it can be implemented efficiently using matrix multiplication.
    Disadvantage: With respect to dot product, multiplicative attention needs a weight matrix to be trained.

Additive attention:

    Advantage: For larger dimensions of decoder states, it performs better than multiplicative attention.
    Disadvantage: With respect to the above two, additive attention mechanism has an additional weight vector whose length is a hyperparameter.
    
 
**2a - Analyzing NMT translation errors**

    i)
     
    ii) 
    
    iii) 
        Source Sentence: Un amigo me hizo eso - Richard Bolingbroke.
        Reference Translation: A friend of mine did that - Richard Bolingbroke.
        NMT Translation: A friend of mine did that - Richard <unk>
        
        Error: Failed to translate the word: Bolingbroke
        Reason: "Bolingbroke" is out-of-vocabulary word as its absent in the training set.
        Solution: Proper nouns should be kept as it is. POS tagging should help in identifying that the last two words are proper nouns. 
        
    iv)
    
    v)
    
    vi) 
        Source Sentence: Eso es m´as de 100,000 hect´areas.
        Reference Translation: That’s more than 250 thousand acres.
        NMT Translation: That’s over 100,000 acres.
        
        Error: Failed to convert numeric value of hectares to acres.
        Reason: Knowledge of units conversion is not used. 
        Solution: Knowledge of units conversion is needed in addition to language translation.
        
**2c - BLEU Score**

    i) 
        Source Sentence s: el amor todo lo puede
        Reference Translation r1: love can always find a way
        Reference Translation r2: love makes anything possible
        NMT Translation c1: the love can always do
        NMT Translation c2: love can make anything possible
        
        Computing BLEU score for c1:
            
            1-grams: "the", "love", "can", "always", "do"
            
            p1 numerator contribution of 1-gram: min( max(count r1(1-gram), count r2(1-gram)), count c(1-gram) )
            
            1-gram: "the"
                    count c("the"): 1
                    count r1("the"): 0
                    count r2("the"): 0
                    p1 numerator contribution: 0
                    
            1-gram: "love"
                    count c("love"): 1
                    count r1("love"): 1
                    count r2("love"): 1
                    p1 numerator contribution: 1
                    
            1-gram: "can"
                    count c("can"): 1
                    count r1("can"): 1
                    count r2("can"): 0
                    p1 numerator contribution: 1
                    
            1-gram: "always"
                    count c("always"): 1
                    count r1("always"): 1
                    count r2("always"): 0
                    p1 numerator contribution: 1
                    
            1-gram: "do"
                    count c("do"): 1
                    count r1("do"): 0
                    count r2("do"): 0
                    p1 numerator contribution: 0
                    
            p1 = (0 + 1 + 1 + 1 + 0)/(5)  = 3/5
                    
                    
            2-grams: ("the", "love"), ("love", "can"), ("can", "always"), ("always", "do")
            
            p2 numerator contribution of 2-gram: min( max(count r1(2-gram), count r2(2-gram)), count c(2-gram) )
            
            2-gram: ("the", "love")
                    count c(("the", "love")): 1
                    count r1(("the", "love")): 0
                    count r2(("the", "love")): 0
                    p2 numerator contribution: 0
                    
            2-gram: ("love", "can")
                    count c(("love", "can")): 1
                    count r1(("love", "can")): 1
                    count r2(("love", "can")): 0
                    p2 numerator contribution: 1
                    
            2-gram: ("can", "always")
                    count c(("can", "always")): 1
                    count r1(("can", "always")): 1
                    count r2(("can", "always")): 0
                    p2 numerator contribution: 1
                    
            2-gram: ("always", "do")
                    count c(("always", "do")): 1
                    count r1(("always", "do")): 0
                    count r2(("always", "do")): 0
                    p2 numerator contribution: 0
                    
            p2 = (0 + 1 + 1 + 0)/4 = 2/4
            
            Brevity Penalty:
                c = 5
                r1: 6 words
                r2: 4 words
                
                r_star: Both r1, r2 are equally close to c. Hence choose the shorter one.
                r_star = 4
                
                Brevity penalty = 1 (as c >= r_star)
                
            BLEU score:
                >>> 1 * np.exp(0.5*np.log(0.6) + 0.5*np.log(0.5))
                0.5477225575051662
            
            
            
        Computing BLEU score for c2:
            
            1-grams: "love", "can", "make", "anything", "possible"
            
            1-gram: "love"
                    count c("love"): 1
                    count r1("love"): 1
                    count r2("love"): 1
                    p1 numerator contribution: 1
                    
            1-gram: "can"
                    count c("can"): 1
                    count r1("can"): 1
                    count r2("can"): 0
                    p1 numerator contribution: 1
                    
            1-gram: "make"
                    count c("make"): 1
                    count r1("make"): 0
                    count r2("make"): 0
                    p1 numerator contribution: 0
                    
            1-gram: "anything"
                    count c("anything"): 1
                    count r1("anything"): 0
                    count r2("anything"): 1
                    p1 numerator contribution: 1
                    
            1-gram: "possible"
                    count c("possible"): 1
                    count r1("possible"): 0
                    count r2("possible"): 1
                    p1 numerator contribution: 1
                    
            p1 = (1+1+0+1+1)/5 = 4/5
            
            2-grams: ("love", "can"), ("can", "make"), ("make", "anything"), ("anything", "possible")
             
            2-gram: ("love", "can")
                    count c(("love", "can")): 1
                    count r1(("love", "can")): 1
                    count r2(("love", "can")): 0
                    p2 numerator contribution: 1
                    
            2-gram: ("can", "make")
                    count c(("can", "make")): 1
                    count r1(("can", "make")): 0
                    count r2(("can", "make")): 0
                    p2 numerator contribution: 0
                    
            2-gram: ("make", "anything")
                    count c(("make", "anything")): 1
                    count r1(("make", "anything")): 0
                    count r2(("make", "anything")): 0
                    p2 numerator contribution: 0
                    
            2-gram: ("anything", "possible")
                    count c(("anything", "possible")): 1
                    count r1(("anything", "possible")): 0
                    count r2(("anything", "possible")): 1
                    p2 numerator contribution: 1
                    
            p2 = (1+0+0+1)/4 = 2/4
            
            Brevity Penalty:
                c = 5
                r1: 6 words
                r2: 4 words
                
                Here also brevity penalty = 1
                 
             BLEU score:
                >>> 1 * np.exp(0.5*np.log(0.8) + 0.5*np.log(0.5))
                0.6324555320336759
                
        
        c2 is considered a better translation as BLEU score of c2 is higher than c1.
        
    ii) Now we only have reference translation r1
    
         Computing BLEU score for c1:
            
            1-grams: "the", "love", "can", "always", "do"
            
            1-gram: "the"
                p1 numerator contribution: 0
                
            1-gram: "love"
                p1 numerator contribution: 1
                
            1-gram: "can"
                p1 numerator contribution: 1
                
            1-gram: "always"
                p1 numerator contribution: 1
                
            1-gram: "do"
                p1 numerator contribution: 0
                    
            p1 = (0+1+1+1+0)/5 = 3/5
            
            
            2-grams: ("the", "love"), ("love", "can"), ("can", "always"), ("always", "do")
            
            2-gram: ("the", "love")
                p2 numerator contribution: 0
                
            2-gram: ("love", "can")
                p2 numerator contribution: 1
                
            2-gram: ("can", "always")
                p2 numerator contribution: 1
                
            2-gram: ("always", "do")
                p2 numerator contribution: 0
                
            p2 = (0+1+1+0)/4 = 2/4
            
            
            Brevity Penalty:
                c = 5
                r1: 6 words
                r_star = 6 (since we now have a single reference translation)
               
                Brevity penalty:
                    >>> np.exp(1 - 6./5)
                    0.8187307530779819
                    
            BLEU score:
                >>> np.exp(1 - 6./5)*np.exp(0.5*np.log(0.6) + 0.5*np.log(0.5))
                    0.448437301984003
            
         
         Computing BLEU score for c2:
        
            1-grams: "love", "can", "make", "anything", "possible"
            
            1-gram: "love"
                p1 numerator contribution: 1
                
            1-gram: "can"
                p1 numerator contribution: 1
                
            1-gram: "make"
                p1 numerator contribution: 0
                
            1-gram: "anything"
                p1 numerator contribution: 0
                
            1-gram: "possible"
                p1 numerator contribution: 0
                
            p1 = (1+1+0+0+0)/5 = 2/5
            
            
            2-grams: ("love", "can"), ("can", "make"), ("make", "anything"), ("anything", "possible")
         
            2-gram: ("love", "can")
                p2 numerator contribution: 1
                
            2-gram: ("can", "make")
                p2 numerator contribution: 0
                
            2-gram: ("make", "anything")
                p2 numerator contribution: 0
                
            2-gram: ("anything", "possible")
                p2 numerator contribution: 0
                
            p2 = (1+0+0+0)/4 = 1/4
                
            Brevity Penalty:
                c = 5
                r1: 6 words
                
                As explained above,
                Brevity penalty:
                    >>> np.exp(1 - 6./5)
                    0.8187307530779819
                    
            BLEU score:
                >>> np.exp(1 - 6./5)*np.exp(0.5*np.log(0.4) + 0.5*np.log(0.25))
                0.25890539701513365
                
                
            c1 has higher BLEU score when we have only r1 as the sole reference translation.
            In my opinion c2 seems better translation.
            Reason: "make anything possible" is semantically closer to "always find a way".
                    Whereas c1 seems to be an incomplete sentence. And "always do" means doing some task.
                    
                    
    iii) Part (ii) shows an instance where single reference translation leads to higher score for the wrong NMT translation.
            Two sentences/phrases can be semantically similar even though there's no overlap of words/phrases.
            So when the NMT transalation has good overlap with the other reference translation which was not provided, but low overlap with the reference translation then this NMT translation is scored less.
              Another issue is that there could be certain part of the sentence which reflects the semantic meaning more than other part(s) of the sentence.
              But BLEU score can't take this into consideration as for a given n, it gives equal weightage of each of the n-grams.
              
    iv) Advantages:
            a. BLEU score is an automated system for evaluation of translations.
            
        Disadvantages:
            a. When a sentence can have multiple translations possible, but not all these translations are provided as reference translation,
                then BLEU score can mislead accuracy of the NMT system.
            