#!/usr/bin/env python
# coding: utf-8

# ## Structure from Ono's Instructions

# In[148]:


import random

instruction = open('yoko_ono_instruction.txt').read()
print(instruction)


# In[149]:


lines_instruction = instruction.split("\n")


# ### Structure 1

# In[150]:


structure_sentence = [lines_instruction[0], lines_instruction[3],lines_instruction[18],lines_instruction[17],lines_instruction[16]]


# In[151]:


print(structure_sentence)


# ### Structure 2

# In[152]:


structure_sentence = [lines_instruction[9], lines_instruction[14],lines_instruction[13],lines_instruction[5],lines_instruction[38]]


# In[153]:


print(structure_sentence)


# ### Structure 3

# In[154]:


structure_sentence = [lines_instruction[32], lines_instruction[35],lines_instruction[29],lines_instruction[23],lines_instruction[41]]


# In[155]:


print(structure_sentence)


# ## Food from Menu

# In[156]:


menu = [
    'smoked ham',
    'nova',
    'scramble',
    'roasted',
    'florentine',
    'sweet lemons',
    'cinnamon',
    'nutella',
    'banana',
    'strawberry',
    'berry',
    'crepe',
    'wine',
    'beer',
    'mimosa',
    'earl grey',
    'english breakfast',
    'coffee',
    'cafe au lait',
    'americano',
    'latte',
    'capuccino',
    'chai',
    'mocha',
    'red rye',
    'cortado',
]


# ## Object from Road

# In[157]:


road = [
    'traffic light',
    'pedestrian crossing',
    'bus stop',
    'restaurant',
    'market',
    'street vendor',
    'monument',
    'memorial',
    'garage',
    'buildings',
    'house',
    'park',
    'road',
    'ally',
    'lane',
    'back street',
    'motorway',
    'red flower',
    'tall tree',
    'concrete jungle',
    'tallest building of neighborhood',
    'ant',
    'dog',
    'rabbit',
    'wind',
    'leaf',
    
]


# ## Direction

# In[158]:


direction = [
    'head south on Jay St toward Willoughby St',
    'go onto Pearl St',
    'slight right',
    'turn onto Boerum Pl',
    'Turn left onto Atlantic Ave',
    '!turn is not allowed Mon-Fri 4:00 - 7:00',
    'slight left',
    'head south 2 blocks',    
]


# ## Poem

# In[159]:


poem = [
    'touch each other',
    'Bury it in the garden and place a marker with a number on it', 
    'Sell it to the rag man',
    'Throw it in the garbage',
    'when all the orchestra members finish counting the stars, or when it dawns.'
    'before dawn Bottle the smell of the room of that particular hour as well.'
    'Cut it and use it as strings to tie gifts with'
    'Find a stone that is your size or weight'
    'Then, let them gradually melt in to the sky'
]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## words from reviews

# In[160]:


import spacy
import tracery
from tracery.modifiers import base_english
from simpleneighbors import SimpleNeighbors


# In[161]:


nlp = spacy.load('en_core_web_md')
lookup = SimpleNeighbors(nlp.vocab.vectors_length)


# In[162]:


text = nlp(open('cafe_review.txt').read())


# In[163]:


words = [item for item in text if item.is_alpha]


# In[164]:


nouns = []
for word in words:
    if word.pos_ == "NOUN":
        nouns.append(word.text)
        lookup.add_one(word.text, word.vector)


# In[165]:


adjs = []
for word in words:
    if word.pos_ == "ADJ":
        adjs.append(word.text)
        lookup.add_one(word.text, word.vector)


# In[166]:


verbs = []
for word in words:
    if word.pos_ == "VERB":
        verbs.append(word.text)
        lookup.add_one(word.text, word.vector)


# In[ ]:





# In[167]:


lookup.build()


# In[168]:


delight =lookup.nearest(nlp('delight').vector)
delicious =lookup.nearest(nlp('delicious').vector)
fun =lookup.nearest(nlp('fun').vector)
think = lookup.nearest(nlp('think').vector)


# ## Alteration Step 1

# In[176]:


rules = {
    "origin": " #direction#,\n Drink poem for group of people,\n Awakening the metropolitan \n When the #menu# is dyed thoroughly in rose \n with the morning light, \n #poem# \n #direction#. \n\n in the morning, At dawn, \n Take the sound of the #road# breathing. \n #verb# a small, almmost invisible, hole in the \n center of the #road# and see the #fun# through it, \n\n #direction#, \n #think# the sound of the #noun# aging. \n #verb# it in the #road# and place a #noun# with a #menu# on it. \n #delight#! Do not tell anybody what you did. Walk all over the city \n #direction# with an empty #noun#. \n\n #poem#, \n 1.on ground 2. in mud 3. in snow 4. on ice 5. in water, \n This should be done in the evening.\n \n #poem# As you wave,\n Imagine one thousand suns in the sky at the same time. ", 

    "adj": adjs,
    "verb": verbs,
    "noun": nouns,
    "delight": delight,
    "delicious": delicious,
    "fun": fun,
    "menu": menu,
    "road": road,
    "direction": direction,
    "think": think,
    "poem": poem
    
}


# In[186]:


grammar = tracery.Grammar(rules)
grammar.add_modifiers(base_english)

new_poem = grammar.flatten('#origin#')
print(new_poem)


# head south on Jay St toward Willoughby St,
#  Drink poem for group of people,
#  Awakening the metropolitan 
#  When the cortado is dyed thoroughly in rose 
#  with the morning light, 
#  !turn is not allowed Mon-Fri 4:00 - 7:00!.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Elaboration With Pipeline

# In[64]:


from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer


# In[65]:


tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
model = AutoModelForCausalLM.from_pretrained('distilgpt2')


# In[66]:


generator = pipeline('text-generation', model=model, tokenizer=tokenizer)


# In[67]:


generator("Drill a small, almmost invisible, hole in the center of the canvas and see the room through it'")


# In[ ]:





# In[ ]:




