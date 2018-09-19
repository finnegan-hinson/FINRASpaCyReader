# FINRASpaCyReader
An experiment to train SpaCy's named entity reader to recognize the company and client names from a FINRA violations PDF. 
Work from my internship at CFA Institute publically shared with permission.

The FINRA violations are public record and can be freely accessed here http://www.finra.org/industry/disciplinary-actions 

In the repository is the single source file, a Python program called SpaCy. The spaCy.py program uses the sample entity provider 
code provided open source by the spaCy project as a base. (Which is available on their GitHub here: 
https://github.com/explosion/spacy/blob/master/examples/information_extraction/entity_relations.py ) All instances of provided 
code are clearly marked throughout.

Also included in the repository is a folder of accuracy testing of the spaCy model on various FINRA violation PDFs. These documents are 
color coded: green words were found as intended; crossed out text was accidentally detected; red text is missing from the named entities 
but present in the PDF document.

This code has been publicly published with the permission of CFA Institute and contains no customer information or data.

Thank you for taking the time to read over my code (and my disclaimers)! The FINRASpaCyReader was my first project in Python.

