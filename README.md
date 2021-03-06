# Medical Concept Normalization

## Challenges:

1. Compound nouns: `Schenkelhalsfraktur - Fraktur des Schenkelhalses`. The main focus of the current work.

2. Possible paraphrases: E66 `Adipositas` has close synonyms: `obesity` or `Fettleibigkeit`. Some words from the concepts database are Latin and some of them have German translations. This situations cannot be inferred

3. Additional context: sometimes precise concept depends on the external conexts, e. g. age of the patient. Example: E66.06 `Adipositas Grad III (WHO) bei Patienten von 18 Jahren und älter, Body-Mass-Index [BMI] von 40 bis unter 50` . In this case, one needs to feed concept normalizer the age of the patient or just normalize to the more general concept "Adipositas"

4. Long phrases, sometimes with additional clauses: E63.1 `Alimentärer Mangelzustand infolge unausgewogener Zusammensetzung der Nahrung`

Related paper: https://aclanthology.org/P19-2055/

## Compound nouns: approaches

### Statistical modelling

Tf-idf-like approach, but on top of BPE encodings from large model or compound noun splitting.