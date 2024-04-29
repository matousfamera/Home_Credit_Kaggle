# 1. Configuration library
  - the variable is "CONFIG_LIBRARY"
  - the keys of the library: datatype= int, value: 0 is for metamodel
  - values of the elements are libraries which represent models
  - the keys of the library are
      "id" - datatype=int, corresponds with the key
      "type" - datatype=string
      "parent" - datatype=int, None for metamodel
      "children" - datatype=list of ints, None for base model
      "make_data_func" - datatype=string, name of the function responsible for creating data for the specific model
      "fit_func" - datatype=string, name of the function responsible for fitting the specific model

# 2. Make data function
    - the name must be specified within the "CONFIG_LIBRARY"
    - the function has no argument
    - the function creates data and saves the dataframe in .parquet file, test data and train data are in the dataframe
    - the dataframe must have a proper format:
      1. It has to contain column "case_id", the datatype of the column is int, every value in the column is unique
      2. It has to contain column "WEEK_NUM", the datatype is int, no nan values
      3. It has to contain column "target" the datatype is int, the value is 0 or 1 or nan.
    - the return datatype is string, the string specifies the address where the .parquet is saved

# 3. Fit function
    - the name must be specified withing the "CONFIG_LIBRARY"
    - the function has ONE argument, the datatype is string, the value is address of the parquet file to use, the parquet file generated with corresponding "make_data_func" is used.
    - the return value is a tuple, the first element of the tuple is class representing the model, the second is dataframe with out-of-fold predictions.
    - the model has to have specific properties:
        1. the class has to contain method predict_proba(X), the argument is Pandas DataFrame which does not include "case_id", "target", "WEEK_NUM"
                  the predict_proba(X) method has to return Numpy Array of the same length as the X, the shape of the np.array must be (len(X), 2), the first value for 0-case, the second for 1-case
    - the second element of the tuple is Pandas Dataframe representing oof predictions
          1. the pd.DataFrame must have column "case_id"
          2. the pd.DataFrame must have column "preds"

      
