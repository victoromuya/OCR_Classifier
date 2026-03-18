
## This project extracts texts from image, pdf files. Classify the files based on the contents. Classification class includes invoice, forms, handwritten, and others. The Dataset used is the RVL - CDIP Dataset downloaded from Kaggle. 

### I have written the deeplearning code with tensorflow in order to improve accuracy

## The pytorch is written to serve as an optional method so the model can better understand:
### 1. Where text appears on the page
### 2. Tables vs paragraphs
### 3. Key-value relationships (perfect for invoices)
### that can be achieved bt using LayoutLM models

### if classifying and extracting only invoice, receipts, I will use using the NER. train a ner model using a dataset of invoices, that tailors this project to a particular domain