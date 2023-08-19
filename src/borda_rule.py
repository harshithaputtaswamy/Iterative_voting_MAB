import itertools
import collections

def borda(ballot):
    n = len([c for c in ballot if c.isalpha()]) - 1
    score = itertools.count(n, step = -1)
    result = {}
    for group in [item.split('=') for item in ballot.split('>')]:
        s = sum(next(score) for item in group)/float(len(group))
        for pref in group:
            result[pref] = s
    return result

def tally(ballots):
    result = collections.defaultdict(int)
    for ballot in ballots:
        for pref,score in borda(ballot).iteritems():
            result[pref]+=score
    result = dict(result)
    return result

ballots = ['A>B>C>D>E',
           'A>B>C=D=E',
           'A>B=C>D>E', 
           ]

print(tally(ballots))