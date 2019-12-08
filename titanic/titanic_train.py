import pandas as pd

from metaflow import FlowSpec, step, IncludeFile


def script_path(filename):
    """
    A convenience function to get the absolute path to a file in this
    tutorial's directory. This allows the tutorial to be launched from any
    directory.
    """
    import os

    filepath = os.path.join(os.path.dirname(__file__))
    return os.path.join(filepath, filename)


class TitanicModeling(FlowSpec):
    """
    Construct a ML model from the Titanic data set.
    - preprocess
    - modeling
    """
    train_data = IncludeFile("train_data",
                             help="The path to Titanic train data file.",
                             default=script_path('train.csv'))

    @step
    def start(self):
        """
        Load train data set
        """
        from io import StringIO

        # Load the data set into a pandas dataaframe.
        df_train = pd.read_csv(StringIO(self.train_data))
        self.X = df_train.drop('Survived', axis=1)
        self.y = df_train['Survived']

        # Compute our two recomendation types in parallel.
        self.next(self.categorical_prep, self.numerical_prep)

    @step
    def categorical_prep(self):
        """
        Preprocessing categorical features
        - Impute missing values
        - One-Hot encoding
        """
        self.categorical_columns = ['Pclass', 'Sex', 'Embarked']

        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import OneHotEncoder
   
        self.cat_imputer = SimpleImputer(strategy='constant',
                                         fill_value='missing')
        self.ohe = OneHotEncoder(handle_unknown='ignore',
                                 sparse=False)
        
        X_imp = self.cat_imputer.fit_transform(self.X[self.categorical_columns])
        X_ohe = self.ohe.fit_transform(X_imp)
        col_name = self.ohe.get_feature_names(
            input_features=self.categorical_columns)
        
        self.X_cat = pd.DataFrame(X_ohe, columns=col_name)
        
        self.next(self.join)
    
    @step
    def numerical_prep(self):
        """
        Preprocessing numerical features
        - Impute missing values with mean
        """
        self.numerical_columns = ['Age', 'SibSp', 'Parch', 'Fare']

        from sklearn.impute import SimpleImputer

        self.num_imputer = SimpleImputer(strategy='mean')
        X_imp = self.num_imputer.fit_transform(self.X[self.numerical_columns])

        self.X_num = pd.DataFrame(X_imp, columns=self.numerical_columns)

        self.next(self.join)
    
    @step
    def join(self, inputs):
        '''
        Concatinate the categorical and numerical columns.
        '''
        X_cat = inputs.categorical_prep.X_cat
        X_num = inputs.numerical_prep.X_num
        self.merge_artifacts(inputs)
        
        self.X_prep = pd.concat([X_cat, X_num], axis=1)

        self.next(self.model_construction)

    @step
    def model_construction(self):
        """
        Construct a Random Forest model.
        """
        from sklearn.ensemble import RandomForestClassifier

        self.rf = RandomForestClassifier(random_state=17)
        self.rf.fit(self.X_prep, self.y)

        self.next(self.end)

    @step
    def end(self):
        """
        Print out the train socre.
        """
        print("RF train accuracy: %0.3f" % self.rf.score(self.X_prep, self.y))


if __name__ == '__main__':
    TitanicModeling()
