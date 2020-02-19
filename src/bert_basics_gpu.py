import tensorflow as tf
import keras
import ktrain
from ktrain import text

########## TO USE GPU ###################################
config =  tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.compat.v1.Session(config=config) 
keras.backend.set_session(sess)
#########################################################

########## TO IGNORE GPU ################
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#########################################

(x_train, y_train), (x_test, y_test), preproc = text.texts_from_folder("../datasets/aclImdb",
                                                                        maxlen=500,
                                                                        preprocess_mode="bert",
                                                                        classes=["pos", "neg"])
                                                                        
learner = ktrain.get_learner(text.text_classifier("bert", (x_train, y_train), preproc=preproc),
                            train_data = (x_train, y_train),
                            val_data = (x_test, y_test),
                            batch_size = 2)

learner.fit_onecycle(2e-5, 1)
predictor = ktrain.get_predictor(learner.model, preproc)
data = ['This movie was horrible! The plot was boring. Acting was okay, though.',
        'The film really sucked. I want my money back.',
        'The plot had too many holes.',
        'What a beautiful romantic comedy. 10/10 would see again!',
        ]
predictor.predict(data)