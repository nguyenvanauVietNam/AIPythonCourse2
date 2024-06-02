# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.
## References

- Code references
https://github.com/Harshalkulkarni92/Create-your-own-Image-Classifier
https://github.com/samerkh/Create-Your-Own-Image-Classifier
https://github.com/topics/image-classifier
https://github.com/Mcamin/Udacity-Image-Classifier
https://github.com/mbshbn/Image-classifier
https://github.com/Mudrikkaushik/Image-Classifier
https://github.com/PaMcD/Create-Your-Own-Image-Classifier
https://github.com/CaterinaBi/udacity-image-classifier
https://github.com/pratik18v/small-image-classifier

- Answer from [Stack Overflow](URL).
https://stackoverflow.com/questions/59126859/image-classifier-always-giving-same-results-out-of-ideas
https://stackoverflow.com/questions/68447960/adding-more-information-than-a-image-to-an-image-classifier-in-keras
https://stackoverflow.com/questions/5703013/large-scale-image-classifier
https://stackoverflow.com/questions/66543159/how-to-have-a-default-output-for-an-image-classifier
https://stackoverflow.com/questions/tagged/image-classification
https://github.com/FirebaseExtended/mlkit-custom-image-classifier/issues/6


If you do not find the flowers/ dataset in the current directory, /workspace/home/aipnd-project/, you can download it using the following commands.

!wget 'https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz'
!unlink flowers
!mkdir flowers && tar -xzf flower_data.tar.gz -C flowers


Running the scripts
To train
python train.py path/to/data --save_dir "/data_dir" --arch vgg13 --learning_rate 0.01 --hidden_units 512 --epochs 20 --gpu

To predict using a trained model
python predict.py path/to/image "/data_dir/checkpoint_resnet50.pth --top_k 3 --category_names cat_to_name.json --gpu
