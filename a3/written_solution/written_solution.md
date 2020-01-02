**1a - Adam Optimizer**
 
    i) As the hyperparameter beta1 is set close to 1, the momentum remains quite close to previous value as there's very little contribution from the gradient of loss for the mini-batch.
        As learning rate is also low, the equation theta <- theta - alpha*momentum makes a very small change to theta.
        This small change as we iterate over the mini-batches ensures low variance.
        
    ii) v -> velocity: rolling average of the magnitude of the gradients
        Update equation of v -> Higher the gradient of loss for the mini-batch, bigger the change of v in that dimension.
        Update equation of theta:  theta <- theta - alpha*momentum/sqrt(v)
            Since sqrt(v) is the denominator,
                Lower the value of v, bigger the change of theta and vice-versa.
                Theta is changed faster when gradient magnitude is low.
                This ensures that in the hyperplane of loss, theta; we allow theta to update faster in the dimensions which have near flat surface.
            This helps by updating theta faster without causing higher variance.
            
            
**1b - Dropout**
    
    i)
    
    ii) Regularization technique is applied during training phase to prevent the parameters from exploding.
    
**2 - Neural Transition-Based Dependency Parsing**

***2a***

Stack;  Buffer;  New dependency;  Transition
---------------------------------------------

[ROOT, parsed, this];   [sentence, correctly];    None;   SHIFT

[ROOT, parsed, this, sentence];  [correctly];   None;  SHIFT

[ROOT, parsed, sentence];   [correctly];    this <- sentence;    None;   LEFT-ARC

[ROOT, parsed];  [correctly];   parsed -> sentence; None;  RIGHT-ARC

[ROOT, parsed, correctly];   [];  None;    SHIFT

[ROOT, parsed];  [];  parsed -> correctly;  RIGHT-ARC

[ROOT];  [];  ROOT -> parsed;  RIGHT-ARC

***2b***

2n steps.
 
n steps for shift to put the words in stack.

n steps for left and right arc tagging which will remove the words from stack.
    
***2f - Dependency Parse errors***

    i)  Error type: erb Phrase Attachment Error
        Incorrect dependency: wedding -> fearing
        Correct dependency: heading -> fearing
    
    ii) Error type: Coordination Attachment Error
        Incorrect dependency: makes -> rescue
        Correct dependency: rush -> rescue
    
    iii)Error type: Prepositional Phrase Attachment Error
        Incorrect dependency: named -> Midland
        Correct dependency: guy -> Midland
    
    iv) Error type: Modifier Attachment Error
        Incorrect dependency: elements -> most
        Correct dependency: crucial -> most
    