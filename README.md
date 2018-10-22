I have dataset "Pima Indians Diabetes Database" in csv format and I will try to predict onset of diabetes based on diagnostic measures.
### File columns:
- **Pregnancies** -- Number of times pregnant
- **Glucose** -- Plasma glucose concentration a 2 hours in an oral glucose tolerance test
- **BloodPressure** -- Diastolic blood pressure (mm Hg)
- **SkinThickness** -- Triceps skin fold thickness (mm)
- **Insulin** -- 2-Hour serum insulin (mu U/ml)
- **BMI** -- Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction** -- Diabetes pedigree function
- **Age** -- Age (years)
- **OutcomeClass** -- variable (0 or 1)

# Order of work
  - read and convert csv data to numpy array
  - process feature scalling with mean normalization(for gradient descent speed up)
  - split data into two sets: training set and cross-validation set
  - process batch/stochastic gradient descent with printing cost on every iteration(for confidence that cost is desceasing on every iteration) for estimating Theta(coefficients)
  - predict outcome on cross-validation set and measure accuracy

# Results
 - Accuracy on cross-validation set is **~81%**
 - Difference of accuracy of stochastic and batch gradient descents is not big -- **~0.5%**  
