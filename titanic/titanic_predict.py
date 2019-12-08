import pandas as pd

from metaflow import FlowSpec, step, IncludeFile, Flow, Run, Parameter


def script_path(filename):
    """
    A convenience function to get the absolute path to a file in this
    tutorial's directory. This allows the tutorial to be launched from any
    directory.
    """
    import os

    filepath = os.path.join(os.path.dirname(__file__))
    return os.path.join(filepath, filename)


class TitanicPredict(FlowSpec):
    """
    Prediction of the test data set.
    The model is taken from the latest successful run
    or the designated run_id.
    """

    run_id = Parameter('id',
                       help="Run execution ID",
                       default='latest_successful')
    test_data = IncludeFile("test_data",
                            help="The path to Titanic test data file.",
                            default=script_path('test.csv'))

    @step
    def start(self):
        """
        Load test data set
        """
        from io import StringIO

        # Load the data set into a pandas dataframe.
        self.X = pd.read_csv(StringIO(self.test_data))

        print('run id: ', self.run_id)
        if self.run_id == 'latest_successful':
            self.train_run = Flow('TitanicModeling').latest_successful_run
        else:
            self.train_run = Run(f'TitanicModeling/{self.run_id}')

        # Compute our two recomendation types in parallel.
        self.next(self.categorical_prep, self.numerical_prep)

    @step
    def categorical_prep(self):
        """
        Preprocessing categorical features
        - Impute missing values
        - One-Hot encoding
        """
        categorical_columns = self.train_run.data.categorical_columns
   
        cat_imputer = self.train_run.data.cat_imputer
        ohe = self.train_run.data.ohe
        
        X_imp = cat_imputer.transform(self.X[categorical_columns])
        X_ohe = ohe.transform(X_imp)
        col_name = ohe.get_feature_names(
            input_features=categorical_columns)
        
        self.X_cat = pd.DataFrame(X_ohe, columns=col_name)
        
        self.next(self.join)
    
    @step
    def numerical_prep(self):
        """
        Preprocessing numerical features
        - Impute missing values with mean
        """
        numerical_columns = self.train_run.data.numerical_columns

        num_imputer = self.train_run.data.num_imputer
        X_imp = num_imputer.transform(self.X[numerical_columns])

        self.X_num = pd.DataFrame(X_imp, columns=numerical_columns)

        self.next(self.join)
    
    @step
    def join(self, inputs):
        '''
        Concatinate the categorical and numerical columns.
        '''
        
        X_cat = inputs.categorical_prep.X_cat
        X_num = inputs.numerical_prep.X_num
        self.train_run = inputs.categorical_prep.train_run
        self.merge_artifacts(inputs)
        
        self.X_prep = pd.concat([X_cat, X_num], axis=1)

        self.next(self.prediction)

    @step
    def prediction(self):
        """
        Predict suvived / not survived
        from the test data set.
        """

        rf = self.train_run.data.rf
        self.y_pred = rf.predict(self.X_prep)

        self.next(self.end)

    @step
    def end(self):
        """
        Save the result.
        """
        df_result = self.X.copy()
        df_result['y_pred'] = self.y_pred

        df_result.to_csv('result.csv', index=None)


if __name__ == '__main__':
    TitanicPredict()
