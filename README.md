# Image Recognition Applied to Waste Classification

 Can we improve our current recycling rates using machine learning?

According to the Environmental Protection Agency (EPA), the difference in 2014 and 2015 effective recycling rates was less than one percent [1]. Also, just 35% of the total waste production in the US was recycled or composted in 2015.
We attribute the slowdown in recycling to the fact that recycling is an effortful and time-consuming task. To explore the use of machine learning we studied a For our project, we chose to utilize a dataset of 2,527 waste images spanning six classes: plastic (482), metal (410), glass (501), cardboard (403), trash (137), and paper (594).
We chose to perform data augmentation which transformed our dataset to reduce overfitting and increase predictive power. These transformations included width and height shift range, flipping, zooming, and brightness changes to increase the data set ten times. 
Once our data was processed we decided on a series of models to fit our dataset and accuracy whose formula is shown below as the best measure of predictive performance. This is because our data set was fairly balanced and our specific application for waste classification calls for equal balancing of the categories. 
We explored the use of CART, Random Forest, Boosting, Neural Network, Convolutional Neural Network, and CNN with transfer learning. 
Our best model gives us an accuracy of 91.3% on new observations. Compared with the actual recycling rates in the US (35%), the high accuracy reveals promising results and potential for adaptability. 
