# df2markov
A simple way to create Markov Chains from dataframes

## Description
df2markov is a simple tool to create a Markov chain from timestamped data. A typical use case is the analysis of web browsing data. df2markov allows you to estimate the transition probabilities of, for instance, reading an article on topic t1 after having read an article about topic t2; or the probabilities of continuing reading an article on a news website after having encountered it on social media. df2markov also allows to plot the transition probabilities.


## Installation
You can either install df2markov from this github repository, or simply install it via pip:
```
pip install df2markov
```

For (optional) plotting, we create [DOT files](https://en.wikipedia.org/wiki/DOT_(graph_description_language)), which can then be converted into various output formats, such as PNG or PS. To make use of this, the `dot` command from [Graphviz](http://graphviz.org) needs to be available on your system. On Ubuntu, you can install it via

```
sudo apt install graphviz
```

For Windows and Mac, have a look at the Graphviz website.


## Usage

As input, df2markov expects your data to be (roughly) organized like this:

| Timestamp  | Session  | User  | State |
| ------------- | ------------- |------------- | ------------- |
| 2019-2-1 13:44:21  | 1  |  Anna   |   C |
| 2019-2-1 13:44:45  | 1  |  Anna   |   A |
| 2019-2-1 13:44:59  | 1  |  Anna   |   A |
| 2019-2-1 13:46:05  | 1  |  Anna   |   F |
| 2019-2-1 17:46:05  | 2  |  Anna   |   A |
| 2019-2-1 17:46:47  | 2  |  Anna   |   F |
| 2019-2-1 13:44:22  | 1  |  Bob    |   D |
| 2019-2-1 13:45:38  | 1  |  Bob    |   D |
| 2019-2-1 13:46:01  | 1  |  Bob    |   F |

df2markov is relatively flexible and accepts different data types in the column of the table: In principle, all columns accept various data types such as integers, floats, strings. 
The only restriction is that the timestamp must be *sortable* in a meaningful way: A simple integer (with increasing values) is fine, as are datetime objects or strings in, for example, ISO 8601 format ("1997-07-16T19:20"). Strings that do not sort in chronological order (e.g., "16-7-1997") would lead to incorrect results.

The session column is optional: it allows you to group data into sessions, such as a web browsing session. For instance, if a user visits website B four hours after visiting website A, you may not want to consider this as a transition.

This is particular useful if one of your states is a (meaningful) absorbing state, such as 'End of Web session'. In that case, one should add a final absorbing state to every session representing the exit point (i.e., a state once entered, cannot be left). In the example, this state is called 'F'. 

However, when one does not have an absorbing state --- for example when examining the weather (which can only be "sunny" or "rainy") --- it is also possible to use the Timestamp column to detect sequence patterns in the data. Again, strings that do not sort in chronological order (e.g., "16-7-1997") would lead to incorrect results.

```
from df2markov import SAMPLEDATA, Markov
```

### Sequential patterns per user:

By creating transition matrices, df2markov organizes the data into meaningful sequential patterns:
```
mymodel = Markov(SAMPLEDATA)
```

Next, df2markov can create a transition probability matrix: the likelihood of transitioning between any two states:

```
mymodel.get_probability_matrices()      
```

It is also able to visualize the transitions:
```
mymodel.plot()      
```

Convert to common graphic formats:
```
dot -Tpng 1_markov_topic.dot > 1_markov_topic.png
```

### Aggregate sequential patterns:
Finally, df2markov is able to aggregate the data from all users in the data set to create an overall (a) percentage matrix (0-100 percent), (b) probability matrix (0-1), or (c) frequency matrix:

```
mymodel.plot(how = "frequency")      
```

## Citation

If you find this package useful and build on it in your academic work, we appreciate the citation of our paper:

Vermeer, S.A.M. & Trilling, D. (2020): Toward a better understanding of news user journeys: A Markov chain approach. *Journalism Studies*, forthcoming.

Bibtex:
```
@df2markov,
author = {Vermeer, Susan A.M. and Trilling, Damian},
title = {Toward a better understanding of news use journeys: A Markov chain approach},
journal = {Journalism Studoes},
year = {2020},
volume = {tbd},
pages = {tbd},
doi = {tbd}
```
