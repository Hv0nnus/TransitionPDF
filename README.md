# TransitionPDF
You are bored of Power Point transition ? Use the first(?) Optimal Transition between your PDF slide based on the Optimal Transport theorie.

WARNING : THIS CODE IS QUITE A MESS FOR THE MOMENT AND USE BOTH PYTHON AND BASH COMMAND.

But in theory, you only have to replace the pdf file Presentation_OT.pdf and run the following command:
python main.py --qualityx 25 --qualityy 25 --qualityx_pres 50 --qualityy_pres 50 --K 100
Then launch pres.html and use keyboard for your presentation.

Use higher value of qualityx/y (250) to increase the quality of transition between slide.
Use higher value of qualityx_pres/y (500) to increase the quality of the slide itself.
I think both of this value are the number of pixel in x and y axis.
K is an hyperparameter of the method used to OT such big images. Basically, it is the number of points when we can use the Optimal Transport. Equivalently it is also the number of centroid for each Kmeans.

