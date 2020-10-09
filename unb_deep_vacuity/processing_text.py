from metaflow import FlowSpec, step, Parameter


def join_relative_folder_path(filename: str) -> str:
    """
    A convenience function to get the absolute path to a file in this
    data's directory. This allows the file to be launched from any
    directory.

    """
    import os

    path_to_data_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data/dic_raw/")
    return os.path.join(path_to_data_folder, filename)


class ProcessingTextFlow(FlowSpec):
    """
    Text Classification for Deep Vacurity

    Summary
    1. Load Data
    2. Exploratory analysis
    3. Data preparation
        3.1. Renaming the columns
    4. Text Filter
        4.1. Removing ponctuation and stopwords
        4.2. Lemmatizing and stemming
        4.3. Entity recognition and filtering
        4.4. Part-of-Speech analysis and filtering
        4.5. Analysing filtering results
    5. Text Parser
        5.1 Dropping missing entities and POS
        5.2 Counting and Vectorizing
        5.3 Grisearching Parameters
    6. Model Train and Cross-Validation
        6.1 Comparing models
        6.2 Evaluating winner model
        6.3 Confusion Matrix
    """
    input_json_path = Parameter(
        "preprocessed_documents_data",
        help="The path to preporcessed documents file.",
        default=join_relative_folder_path('dic_raw_0_1.json'))

    target_variable_name = Parameter(
        "target_variable_name",
        help="The target variable label",
        default='RISCO')

    text_variable_name = Parameter(
        "text_variable",
        help="The text variable label",
        default='TXT')

    @step
    def start(self):
        """
        1. Load data
        We'll have a look at the raw data and analyse it's structure.
        """
        import pandas as pd
        from io import StringIO
        from datetime import datetime
        self.start_time = datetime.now()
        # This will be used to load spacy model
        self.vector_model_name = "pt_core_news_sm"
        # Load the data set into a pandas dataframe.
        print(self.input_json_path)
        self.dataframe = pd.read_json(
            "/home/dev/algoritmos/UnB/Deep Vacurity/unb_deep_vacuity/unb_deep_vacuity/resources/data/dic_raw/dic_raw_0_1.json", orient='records')
        print(self.dataframe.head())
        self.next(self.exploratory_analysis)

    @step
    def exploratory_analysis(self):
        self.next(self.end)

    @step
    def end(self):
        """
        End the flow.
        Output data available on
            mapped_distances_df
        """
        from datetime import datetime
        
        self.end_time = datetime.now()
        self.total_time = self.end_time - self.start_time
        pass


if __name__ == '__main__':
    ProcessingTextFlow()
