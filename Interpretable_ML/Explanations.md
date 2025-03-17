- According to [Wikipedia](https://en.wikipedia.org/wiki/Explanation), An explanation is a set of statements usually constructed to describe a set of facts that clarifies the causes, context, and consequences of those facts.
- But, What is a "good" explanation? This article is based on [Miller (2017)](https://arxiv.org/abs/1706.07269) and [Interpretable ML by Christoph Molnar](https://christophm.github.io/interpretable-ml-book/interpretability.html). 

# What is a good explanation?
- An explanation is an answer to **why-question**.
	- Why did the Tesla stock went down?
	- Why was my loan rejected?

### Explanations are contrastive
- We are not generally interested in "why" a certain decision was made, but what would have caused this decision? What would have happened in **"that"** scenario?
- We like more counterfactual explanations than original explanation. Say, If my house loan was rejected, I am not interested in all the factors which caused my loan rejection instead I would be more interested in what would have happened if my factors were like **"this"**. 
- We like to know the contrast between 