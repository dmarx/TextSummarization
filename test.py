from summarizer import summarize

#with open('article.txt', 'r') as f:
with open('article2.txt', 'r') as f:
    text = f.read()
    
test = summarize(text)
