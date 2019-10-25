# Parameter identifiability example

This example demonstrates the consequences of unidentifiable parameters.
![Figure 2](herg-fig/herg-fits-and-predictions.png)

### Requirements
The code requires Python 2.7+ or 3.5+ and one dependency: [PINTS](https://github.com/pints-team/pints).

### Steps to reproduce the figure/example
1. Run fits using `herg-fit-ap.py` and `herg-fit-staircase.py` with argument `[int:fit_id]` for different random seeds and initial guesses.
2. Run `herg-rank-ap.py` and `herg-rank-staircase.py` to rank the obtained parameters in step 1.
3. Run `herg-predict.py` to plot and compare the predictions using different voltage protocols.

### Other files
- `herg.py`: The hERG model.
- `ap-protocol.csv`: An action potential voltage protocol.
- `staircase-protocol.csv`: The staircase voltage protocol from <https://github.com/CardiacModelling/hERGRapidCharacterisation>.
