# Unknown user handling for setting 3

For systems which use setting 3, the following describes how the RS handles the unseen user/item in the test set.

## How the analysis was done

1. Tracing from the point of dataset split into train/test set
2. How does the model obtain the set of users it will use for testing
3. How does the model verify its prediction based on ground truth

## Table summary of findings

| RecSys   | Rec list generation from only users in test set | Rec list generation using all users in train and eval using test set | Similarity matrix only users from train set | Similarity matrix with user from entire dataset | Unknown user ignored | Data leakage |
| :------- | :---------------------------------------------- | :------------------------------------------------------------------- | :------------------------------------------ | :---------------------------------------------- | :------------------- | :----------- |
| DaisyRec | ✅                                              |                                                                      |                                             | ✅                                              |                      | ✅           |
| Elliot   |                                                 | ✅                                                                   | ✅                                          |                                                 | ✅                   |              |
| RecPack  |                                                 | ✅                                                                   | ✅                                          |                                                 | ✅                   |              |