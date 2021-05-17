# psofs
Particle swarm optimization for feature selection


An example:
```r
library(caret)
library(doMC)
library(ggplot2)
library(plyr)

source('psofs.R')

training.x = ...
training.y = ...
testing.x = ...
testing.y = ...

# Similar to the x2index but with a cutoff of 0.7
smallerSubsets <- function(x) x2index(x, 0.7)

pso <- psofs(x = training.x, y = training.y, 
             tx = testing.x, ty = testing.y,
             convert = smallerSubsets,
             functions = psoGA,
             control = list(maxit=200))


g <- plot.psofs(pso)
g <- g + ylim(0.5, 1)
print(g)

```
