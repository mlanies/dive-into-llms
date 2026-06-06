```python
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random
from copy import deepcopy
# Constants
EOS = 50256  # GPT-2's EOS token ID
UNK = 50256  # Using EOS as UNK since GPT-2 doesn't have a specific UNK token

# Utils
def save_text(texts, path):
    with open(path, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    Cell In[5], line 6
          4 import random
          5 from copy import deepcopy
    ----> 6 from scripts.Constants import EOS, UNK
          7 from scripts.utils import save_text
    

    ModuleNotFoundError: No module named 'scripts.Constants'



```python

class Node:
    def __init__(self, freq):
        self.left = None
        self.right = None
        self.father = None
        self.freq = freq
    def isLeft(self):
        return self.father.left == self

def createNodes(freqs):
    return [Node(freq) for freq in freqs]

def createHuffmanTree(nodes):
    queue = nodes[:]
    while len(queue) > 1:
        queue.sort(key=lambda item:item.freq)
        node_left = queue.pop(0)
        node_right = queue.pop(0)
        node_father = Node(node_left.freq + node_right.freq)
        node_father.left = node_left
        node_father.right = node_right
        node_left.father = node_father
        node_right.father = node_father
        queue.append(node_father)
    queue[0].father = None
    return queue[0]

def huffmanEncoding(nodes, root):
    codes = [''] * len(nodes)
    for i in range(len(nodes)):
        node_tmp = nodes[i]
        while node_tmp != root:
            if node_tmp.isLeft():
                codes[i] = '0' + codes[i]
            else:
                codes[i] = '1' + codes[i]
            node_tmp = node_tmp.father
    return codes

class Huffman():
    def __init__(self, k=None, bits=None):
        self.k = k
        self.bits = bits
        
    def extract(self, pred_prob, word_index):
        bit = None
        prob, idx = torch.topk(pred_prob, k=self.k, dim=1)
        word_prob = [[i.item(),j.item()] for i,j in zip(idx[0], prob[0])]
        nodes = createNodes([item[1] for item in word_prob])
        root = createHuffmanTree(nodes)
        codes = huffmanEncoding(nodes, root)
        for i, w_p in enumerate(word_prob):
            if w_p[0] == word_index:
                bit = codes[i]
                break
        return bit
    
    def __call__(self, pred_prob):
        prob, idx = torch.topk(pred_prob, k=self.k, dim=1)
        word_prob = [[i.item(),j.item()] for i,j in zip(idx[0], prob[0])]
        nodes = createNodes([item[1] for item in word_prob])
        root = createHuffmanTree(nodes)
        codes = huffmanEncoding(nodes, root)
        for i,code in enumerate(codes):
            bit = self.bits[:len(code)]
            if bit == code:
                bit_word_index = word_prob[i][0]
                self.bits = self.bits[len(code):]
                break
        if self.extract(pred_prob,bit_word_index)!=bit:
            print("False")
        bit_word_index = torch.LongTensor([bit_word_index]).to(pred_prob.device)
        return bit_word_index

class FLC():
    def __init__(self, k=None, bits=None):
        self.k = k
        self.bits = bits
    
    def extract(self, pred_prob, word_index):
        bit = None
        prob, idx = torch.topk(pred_prob, k=2**self.k, dim=1)
        word_prob = [[i.item(),j.item()] for i,j in zip(idx[0], prob[0])]
        codes = [str(bin(i))[2:].zfill(self.k) for i in range(2**self.k)]
        for i, w_p in enumerate(word_prob):
            if w_p[0] == word_index:
                bit = codes[i]
                break
        return bit
    
    def __call__(self, pred_prob):
        prob, idx = torch.topk(pred_prob, k=2**self.k, dim=1)
        word_prob = [[i.item(),j.item()] for i,j in zip(idx[0], prob[0])]
        codes = [str(bin(i))[2:].zfill(self.k) for i in range(2**self.k)]
        bit = self.bits[:self.k]
        for i,code in enumerate(codes):
            if bit==code:
                bit_word_index = word_prob[i][0]
                self.bits = self.bits[self.k:]
                break
        bit_word_index = torch.LongTensor([bit_word_index]).to(pred_prob.device)
        return bit_word_index

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def generate_text_with_steganography(model, tokenizer, prompt, handle, max_length=30):
    device = model.device
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate steganographic text
    stega_output = input_ids.clone()
    normal_output = input_ids.clone()
    
    for _ in range(max_length):
        if len(handle.bits) == 0:
            break
            
        # Get model predictions
        outputs = model(stega_output)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_probs = F.softmax(next_token_logits, dim=-1)
        
        # Apply steganography
        bit_word_index = handle(next_token_probs)
        stega_output = torch.cat([stega_output, bit_word_index.unsqueeze(1)], dim=1)
        
        # Generate normal text
        outputs = model(normal_output)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(next_token_probs, num_samples=1)
        normal_output = torch.cat([normal_output, next_token], dim=1)
        
        if next_token[0].item() == tokenizer.eos_token_id:
            break
    
    return stega_output, normal_output

```


```python

def main():
    setup_seed(2023)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load GPT-2 model and tokenizer
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    
    # Generate random bits
    bits = ''.join([str(random.choice(list(range(2)))) for i in range(3000)])
    
    # Initialize steganography handler
    k = 2
    handle = Huffman(k=2**k, bits=bits)
    # handle = FLC(k=k, bits=bits)
    
    # Generate text with steganography
    prompt = "Once upon a time"
    stega_output, normal_output = generate_text_with_steganography(
        model, tokenizer, prompt, handle
    )
    
    # Decode and save results
    stega_text = tokenizer.decode(stega_output[0], skip_special_tokens=True)
    normal_text = tokenizer.decode(normal_output[0], skip_special_tokens=True)
    
    save_text([stega_text], './outputs-gpt2-stega.txt')
    save_text([normal_text], './outputs-gpt2-normal.txt')

if __name__ == '__main__':
    main() 
```
