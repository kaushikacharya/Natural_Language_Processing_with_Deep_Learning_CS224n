**1f - Sentence Masks**

Attention scores are computed by taking dot product of decoder hidden vector with projected encoder hidden vector. Vector products are computed over batch of sentences. As these sentences may have variable word length, padded tokens are added to shorter sentences.
 
Masks are used to ensure zero contribution to attention scores from these padded tokens.
e_t values corresponding to padded tokens are assigned -Inf. 
As exp(-inf)=0, hence alpha_t (=softmax(e_t)) gets zero for these padded tokens.
 