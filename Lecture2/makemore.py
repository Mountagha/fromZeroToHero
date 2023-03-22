"""
you give this script some words (one per line) and it will generate more things like it.
uses super state of the art Transformer AI tech
this code is intended to be super hackable. tune it to your needs.

Changes from minGPT:
- I removed the from_pretrained function where we init with GPT2 weights
- I removed dropout layers because the models we train here are small,
  it's not necessary to understand at this stage and this scale.
- I removed weight decay and all of the complexity around what parameters are
  and are not weight decayed. I don't think this should make a massive 
  difference at the scale that we operate on here.
"""