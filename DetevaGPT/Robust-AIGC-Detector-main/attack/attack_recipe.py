from textattack import Attack
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.pre_transformation import MaxModificationRate
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.goal_functions import InputReduction, UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import WordDeletion, WordSwapWordNet

from textattack.attack_recipes import AttackRecipe


class InputReductionFeng2018(AttackRecipe):
    """Feng, Wallace, Grissom, Iyyer, Rodriguez, Boyd-Graber. (2018).

    Pathologies of Neural Models Make Interpretations Difficult.

    https://arxiv.org/abs/1804.07781
    """

    @staticmethod
    def build(model_wrapper):
        # At each step, we remove the word with the lowest importance value until
        # the model changes its prediction.
        transformation = WordDeletion()

        constraints = [RepeatModification(), StopwordModification()]
        #
        # Goal is untargeted classification
        #
        goal_function = InputReduction(model_wrapper, maximizable=True)
        #
        # "For each word in an input sentence, we measure its importance by the
        # change in the confidence of the original prediction when we remove
        # that word from the sentence."
        #
        # "Instead of looking at the words with high importance values—what
        # interpretation methods commonly do—we take a complementary approach
        # and study how the model behaves when the supposedly unimportant words are
        # removed."
        #
        search_method = GreedyWordSwapWIR(wir_method="delete")

        return Attack(goal_function, constraints, transformation, search_method)

class PWWSRen2019_threshold(AttackRecipe):
    """Add threshold
    """

    @staticmethod
    def build(model_wrapper, target_max_score=None):
        transformation = WordSwapWordNet()
        constraints = [RepeatModification(), StopwordModification()]
        goal_function = UntargetedClassification(model_wrapper, target_max_score=target_max_score)
        # search over words based on a combination of their saliency score, and how efficient the WordSwap transform is
        search_method = GreedyWordSwapWIR("weighted-saliency")
        return Attack(goal_function, constraints, transformation, search_method)