### Hyperparamters used for Tests

## Test 0 (Intial tuning)

- Loss function → BCE loss
- Value of positive weight - 3
- Predictive threshold - 0.5
- Image resolution - 256 x 256
- Epochs - 40

## Test 1
- Loss function → BCE loss with positive weight + Dice loss
- Value of positive weight - 3
- Predictive threshold - 0.5
- Image resolution - 384 x 384
- Max Epochs - 40

## Test 2
- Loss function → BCE loss with positive weight + Dice loss
- Value of positive weight - 3
- Predictive threshold - 0.5
- U-net input resolution - 384 x 384
- Max Epochs - 100
- Early stopping patience - 80

## Test 3
- Loss function → BCE loss with positive weight + Dice loss
- Value of positive weight for BCE loss - 3
- Predictive threshold - 0.5
- U-net input resolution - 384 x 384
- Epochs - 150
- Early stopping patience - 130

