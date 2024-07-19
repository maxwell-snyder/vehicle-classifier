# Vehicle Classifier Project

This is my first project. I am 17 years old and have little to no coding experience. I used ChatGPT to help me with the idea, model, data, app, and website. It also taught me the basics of Python, VS Code, Heroku, GitHub, and how to use ChatGPT in general, to which I previously had no experience using. This project is a vehicle classifier that will look at most types of image files and predict whether it is a Sedan, SUV, or Truck. This is the link to the website I made to use the app: [Vehicle Classifier](https://vehicle-classifier-c3f7ed58873f.herokuapp.com/)

## How to Use

1. **Download Visual Studio Code**
   - Download Visual Studio Code from the website. Set up your account and download all necessary files.
   - Open VS Code, go to the top left, click on `File`, then `Open File`, and select the file you want to open (vehicle-classifier).

2. **Download and Resize Images**
   - Download images that you want to classify (three different categories).
   - Use the `resize_images.py` file to resize all the images to the correct size (150px x 150px). Ensure that the path to the file is set correctly for this and all future programs.
   - Run the program by clicking the play button in the top right that says `Run Python File`. Check the output to ensure the images have been resized correctly.

3. **Split the Data**
   - Split the data (80% training / 20% validation) using the `dataset_splitter.py` file.
   - Verify everything is correct, run the file, and check to ensure the data has been split correctly.

4. **Train the Model**
   - Open the `vehicle_classifier_model.py` file, make sure the paths are set to the correct files, and run the file.

5. **Test the Model**
   - Go to the `image_tester.py` file, add the path to the image you want to test, and make sure it is loading the correct file for the model.
   - Run the program to test the model. The model may not be perfect, but you can always add more data to improve its accuracy.

6. **Quantize the Model**
   - If you want to quantize the model, use the `quantize_model.py` file. Ensure it is quantizing the correct model, then run it.
   - Go back into the `image_tester.py` file and use the quantized model to verify that it works.

7. **Push to Heroku**
   - Download Heroku and create an account.
   - Go to the `app.py` file, change the model to the quantized one.
   - Push it to Heroku using this command in the terminal (which you can access by pressing `Ctrl+``):

   ```sh
   git push heroku master

 - At the end, it should give you a link to a website that looks like mine but is using your data.
If you get an error message, you can put it into ChatGPT with context, and it should help you work through it or assist you in redesigning the website in the index.html file.

