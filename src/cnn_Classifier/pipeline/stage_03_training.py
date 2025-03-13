from cnn_Classifier.config.configuration import CofigurationManager
from cnn_Classifier.components.prepare_callbacks import PrepareCallback
from cnn_Classifier.components.training import Training
from cnn_Classifier import logger


STAGE_NAME = "Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = CofigurationManager()
        prepare_callbacks_config = config.get_prepare_callback_config()
        prepare_callbacks = PrepareCallback(config=prepare_callbacks_config)
        callback_list = prepare_callbacks.get_tb_ckpt_callbacks()

        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()  # Load the updated base model into the training instance
        training.train_valid_generator()
        training.train(
            callback_list=callback_list
        )


if __name__ == "__main__":
    try:
        logger.info(f"************************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx=========x")
    except Exception as e:
        logger.exception(e)
        raise e