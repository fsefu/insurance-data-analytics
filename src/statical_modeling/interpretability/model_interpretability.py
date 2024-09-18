import shap
import lime
import lime.lime_tabular

class ModelInterpretability:
    def __init__(self, model, X_test):
        self.model = model
        self.X_test = X_test

    def shap_analysis(self):
        """SHAP analysis to interpret the model."""
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X_test)
        shap.summary_plot(shap_values, self.X_test)

    def lime_analysis(self):
        """LIME analysis for model interpretability."""
        explainer = lime.lime_tabular.LimeTabularExplainer(self.X_test.values, mode="regression", feature_names=self.X_test.columns)
        i = 0  # Analyze the first instance in the test set
        exp = explainer.explain_instance(self.X_test.values[i], self.model.predict, num_features=5)
        exp.show_in_notebook(show_table=True)
