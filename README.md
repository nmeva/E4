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

### *1.This code block shows 3 LSTM Layers created: (previous code is commented)*
``` python

        # self.rnn = nn.LSTM(embedding_dim, 
        #                    hidden_dim, 
        #                    num_layers=n_layers, 
        #                    bidirectional=bidirectional, 
        #                    dropout=dropout)

        self.lstms = nn.ModuleList()
        self.lstms.append(nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout))
        self.lstms.append(nn.LSTM(hidden_dim*2, 
                    hidden_dim, 
                    num_layers=n_layers, 
                    bidirectional=bidirectional, 
                    dropout=dropout))
        self.lstms.append(nn.LSTM(hidden_dim*2, 
            hidden_dim, 
            num_layers=n_layers, 
            bidirectional=bidirectional, 
            dropout=dropout))

```

### *2.This code shows the "for" loop being used in the "forward" function:*

#### Old pack sequence:
``` python
        #pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)		   
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
```        
#### Updated pack sequence:
``` python
        #pack sequence
        packed_output = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        
        for lstm in self.lstms:
          packed_output, (hidden, cell) = lstm(packed_output)
```


### *3.This code block shows the DROPOUT variable, we can set it to 0.2:
We can do that before calling the model or while calling the model:(previous code is commented)

``` python
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
#DROPOUT = 0.5
DROPOUT = 0.2
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = RNN(INPUT_DIM, 
            EMBEDDING_DIM, 
            HIDDEN_DIM, 
            OUTPUT_DIM, 
            N_LAYERS, 
            BIDIRECTIONAL, 
            DROPOUT, 
            PAD_IDX)
````


### *4.This code block helps in reversing text for both train and test:*
``` python
for _ in range(len(train_data)):
  vars(train_data.examples[_]).get('text').reverse()

for _ in range(len(test_data)):
  vars(test_data.examples[_]).get('text').reverse()
```

### *5.The final test accuracy is *87%* after running for 15 epochs*
``` python
model.load_state_dict(torch.load('tut2-model.pt'))
test_loss, test_acc = evaluate(model, test_iterator, criterion)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.0f}%')
````
Test Loss: 0.315 | Test Acc: 87%

### [6.Colab File Link](https://colab.research.google.com/drive/1EaY67rczy0g9ib6staw7ljH1Jagb09Bt?usp=sharing)
