# What is the fundamental difference between GPT and BERT?

It lies in which side of the context each position can see; their training objectives and uses follow from this. The [[self-attention]] of [[GPT]] carries a [[causal mask]], so position $i$ sees only $1, \dots, i$; trained to predict the next [[token]], it is a [[language model]] and naturally suited to generating text word by word. The self-attention of [[BERT]] has no mask, so each position sees both sides at once, and it is trained to predict masked tokens. It gives each token a representation informed by the whole sentence, which suits understanding tasks such as classification and extraction, but it cannot directly generate text word by word.

Structurally, GPT uses only the [[decoder]] layers of the [[Transformer]], without [[cross-attention]], and BERT only the [[encoder]] layers. Inside, the layers of the two are nearly identical; the difference lies only in that mask.

Example: In “the cat sleeps”, GPT's representation of “cat” is derived from “the” and “cat” alone, whereas BERT's is derived from “the”, “cat” and “sleeps” together.
