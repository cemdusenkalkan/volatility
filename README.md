# volatility
volatility forecasting.

first, i want to share the issues I met while doing this project, the biggest part to solve is data handling, and it was sure the hardest too. You can find the similar codes in internet while doing a project like this, that was what I did at least, I consider myself as a beginner in machine learning python project, actually while writing this, I don't even know about the success of code, not the model itself.
I still can not run it. Meanwhile, to take a break from solving the NaN Value errors from RandomForestRegressor. I hope I can solve it.


Maybe checking the values with ifnull can solve this problem.

It solved! But now I encounter those nanvalues at log file? however, even running this code is a great job for me. I am sure I will handle them too

-Maybe those issues can appear at runtime while making some feature engineering, and these calculations can lead to NaN data, as we see, we have no NaNs at csv file, so problem must be like that.

<img width="998" alt="image" src="https://github.com/cemdusenkalkan/volatility/assets/99793829/3ecbddd4-0676-49e7-bcd9-c83a62aa92e4">

IT WORKED ! ChatGPT found a killer solution (this might make so much valuable data loss and can surely affect the performance in the bad way, but for now, it works!)

Since it is finished, I will upload the first running version of this code and try to make some updates in future.

