from cnn_Classifier.config.configuration import CofigurationManager
from cnn_Classifier.components.evaluation import Evaluation
from cnn_Classifier import logger

STAGE_NAME = "Evaluation stage"

class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = CofigurationManager()
        val_config = config.get_validation_config()
        evaluation = Evaluation(config=val_config)
        evaluation.evaluation()
        evaluation.save_score()



if __name__ == "__main__":
    try:
        logger.info(f"************************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx=========x")
    except Exception as e:
        logger.exception(e)
        raise e