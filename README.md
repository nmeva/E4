# Session 4 Assignment

## [Reference Starting File](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb)

## *Assignment:* Change the above File in such a way that:

1. it has 3 LSTM layers
2. it has used a for loop to do so in the forward function
3. the dropout value used is 0.2
4. trained on the text that is reversed (for example "my name is Rohan" becomes "Rohan is name my"
5. achieves 87% or more accuracy
6. once done, share the Github link as well (after training on Google Colab, move the file to GitHub).


### The following changes were made:

### *1.This code block shows 3 LSTM Layers created:*
``` python

```

### *2.This code shows the "for" loop being used in the "forward" function:*
``` python

```


### *3.This code shows the dropout variable created and value set to "0.2":*
``` python

````


### *4.This code helps in reversing text for both train and test:*
``` python
for _ in range(len(train_data)):
  vars(train_data.examples[_]).get('text').reverse()

for _ in range(len(test_data)):
  vars(test_data.examples[_]).get('text').reverse()
```

### *5.The final test accuracy is *87%* after running for 15 epochs*
``` python

````

### *6.Colab File Link*
