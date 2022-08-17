# Report
A larger overview of what I've already done so far. 

# Part 1: Classifying the attributes
In the original thesis, there where some columns already containing categorical information. I started by trying to predict those columns myself. The code for this can be found in `1 Label classification.ipynb`

## Approach 1: TF-IDF & MultinomialDB
The problem with this approach is that it almost always predicts the same class (also because of a data inbalance).

## Approach 2: Simple Rules
Some of the attributes have a clear definition (e.g. "Length" is true for any sentence having more than 76 characters). Implemently those rules directly, will give more accurate results.
This works of course better. There were still some wrong classifications, but this is sometimes because of human error in the original label (e.g. in the case of missclassification with length) or because some other semantical issues like retorical questions that aren't always seen as normal questions. 
This gives quite good results for "Lang", "Vragen", "Cijfers", "Citaat" and moderate results for "Interpunctie". Results for "Lidwoorden" where worse than expected (because I didn't look into whether you could drop the article or not).

## Overview per attribute
### Actief
Tried predicting based on forms of "worden/zijn", "door" or the prefix "ge-" (for detecting a past particle), but no good results. 
Main problem is probably in past particle prediction. Tried different thinks, which became slightly better, but no huge differences. 
### Lang
Good results, because of the clear definition of this attribute (length > 76).
### Vragen
Good results by simply checking the presence of a question mark
### Interpunctie
Moderate results by just checking the different interpunction characters. 
### Tweeledigheid
Good results by just checking for a ":" or "." that split the sentence in two parts. 


# Manual labeling
Gekeken naar Cross-Encoders/Bi-Encoders (in sentence-transformers library), maar deze waren vooral om gerelateerde zinnen met elkaar te gebruiken (bv. om een tegenspraak tussen 2 zinnen te vinden, ipv een winner te selecteren)
Gekeken naar content-based recommendation manieren, maar ook hier het probleem dat dit soort recommendation helemaal anders werkt (we willen niet van heel de itemset relevante items recommenden, maar steeds van de candidate headlines telkens eentje aanraden)

