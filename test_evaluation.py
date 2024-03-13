import pytest
from deepeval import assert_test
from deepeval.metrics import HallucinationMetric#, UnBiasedMetric, NonToxicMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langchain_community.llms import LlamaCpp
import yaml

prompt_template="""
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Generate a cooking recipe using the following ingredients:
Ingredients: {user_input}

Only return the helpful recipe below and nothing else.
Helpful recipe:
"""

dataset = None

with open("output.yaml", "r") as s:
    dataset = yaml.safe_load(s)

hallucination_metric = HallucinationMetric(threshold=0.3)
#unbiased_metric = UnBiasedMetric(evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT], threshold=0.5)
#non_toxic_metric = NonToxicMetric(evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT], threshold=0.5)

llm = LlamaCpp(model_path="model/recipe_ingrident.Q4_K_M.gguf", n_batch=512, temperature=0.9)

@pytest.mark.parametrize(
    "sample_case",
    dataset,
)
def test_case(sample_case: dict):
    user_input = sample_case.get("question", None)
    expected_output = sample_case.get("answer", None)
    context = sample_case.get("context", "")  

    actual_output = llm(prompt_template.format(user_input=user_input))

    test_case = LLMTestCase(
        input=user_input,
        actual_output=actual_output,
        expected_output=expected_output,
        context=list(context),
    )

    assert_test(test_case, [hallucination_metric])#, non_toxic_metric, unbiased_metric])
