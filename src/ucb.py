# Warmup stage:
# The algorithm starts by playing each arm once. 

# Continuing stage
# At each time stept, UCB plays the arma hat maximizes < ra>+√2 lntta, 
# where< ra>is the averagereward obtained from arma, andtais the number 
# of times armahas been playing so far.