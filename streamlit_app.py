import streamlit as st
import pandas as pd
import numpy as np
#for displaying images
from PIL import Image
import seaborn as sns
import codecs
import streamlit.components.v1 as components

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

left_co, cent_co,last_co = st.columns(3)

st.title("Lebron's Game Points Prediction")

image_path = Image.open("nba-lebron-james-record-milliard-fortune-cigare.webp")

st.image(image_path,width=400)

app_page = st.sidebar.selectbox("Select Page",['Business Case','Data Exploration','Visualization','Prediction', 'Data Insights'])

df = pd.read_csv("lebron-game-log-dataset.csv")

if app_page == 'Business Case':

    st.title("1. Business Case")

    st.subheader("Objective:")
    
    st.write("The purpose of this dashboard is to analyze LeBron James’ game performance across 3 different seasons (2021-2023) and explore potential relationships between his game stats and his game points. For this page, we aim to create an efficent model using Linear Regression that will help us predict LeBron’s stats in his upcoming seasonal games in future by examining historical trends and the relationship between key performance variables.")

    st.subheader("Key Questions:")

    st.write("""
            - How have LeBron’s key statistics (points, rebounds, assists, etc.) evolved over his career?
            - What patterns emerge in his game performance during each season?  
            - Can we establish relationships between variables like minutes played, shooting percentage, and points scored to create a model for theoretical projections of his future performance?
            """)

if app_page == 'Data Exploration':
    
    st.title("2. Data Exploration")

    st.dataframe(df.head(5))

    st.subheader("01 Description of the dataset")

    st.dataframe(df.describe())

    st.write("The dataset used contains LeBron's game stats from his previous games of the past 3 seasons season (e.g., points, rebounds, assists, minutes played, etc.).")

    if st.button("Show table key"):
        
        st.write("""Date: The game’s date might capture trends like fatigue, rest days, or performance fluctuations over time (e.g., based on the phase of the season or back-to-back games).
        Opp (Opponent): The strength of the opponent's defense can significantly impact scoring. Historical performance against certain teams can also be factored in.
        Score: This indicates whether the game was a high- or low-scoring affair, which could relate to LeBron's contribution to the team's points.
        Type (Home/Away): Whether the game is played at home or away can influence performance due to factors like crowd support, travel fatigue, etc.
        Min (Minutes Played): More playing time typically correlates with higher points.
        FGM (Field Goals Made): Direct measure of successful shots, and thus directly correlates with points scored.
        FGA (Field Goals Attempted): Indicates shot volume, which gives insight into LeBron's involvement in the offense.
        FG% (Field Goal Percentage): Efficiency from the field. A higher FG% usually corresponds to higher points.
        3PM (3-Pointers Made): These shots are worth more points, so successful 3-point shots have a direct impact on the total.
        3PA (3-Point Attempts): Reflects how much LeBron is relying on 3-point shots. More attempts can lead to more points if his accuracy holds.
        3P% (3-Point Percentage): Efficiency from 3-point range can help explain scoring variance.
        FTM (Free Throws Made): Directly contributes to points scored.
        FTA (Free Throws Attempted): More attempts suggest more opportunities to score from the line, often due to fouls.
        FT% (Free Throw Percentage): Efficiency at the free-throw line. Higher FT% leads to more points from free throws.
        Rebounds (OR, DR, Reb): Rebounding (offensive and defensive) may not directly predict points but can indicate possession control and second-chance points (for offensive rebounds).
        Ast (Assists): While assists don't directly contribute to his points, they can indicate LeBron's involvement in the game’s offense, which might correlate with higher overall involvement and scoring.
        TO (Turnovers): High turnovers could indicate a game where LeBron struggled, potentially affecting his points.
        Stl (Steals), Blk (Blocks): Defensive stats that might reflect overall game activity, but their direct impact on points is less clear.
        PF (Personal Fouls): More fouls might lead to reduced playing time or aggressive play, affecting point production.
        +/- (Plus-Minus): Reflects the team's performance with LeBron on the court but may not be a strong predictor of his individual points.
        Pts (Points): This is the target variable you want to predict.""")

    st.subheader("02 Missing values")

    dfnull = df.isnull()/len(df)*100
    total_missing = dfnull.sum().round(2)
    st.write(total_missing)

    if total_missing[0] == 0.0:
        st.success("Congrats you have no missing values")

    if st.button("Generate Report"):

        #Function to load HTML file
        def read_html_report(file_path):
            with codecs.open(file_path,'r',encoding="utf-8") as f:
                return f.read()

        # Inputing the file path 
        html_report = read_html_report("report.html")

        # Displaying the file
        st.title("Streamlit Quality Report")
        st.components.v1.html(html_report,height=1000,scrolling=True)

if app_page == 'Visualization':
    st.title("3. Data Visualization")

    df['Month'] = pd.to_datetime(df['Date'] + ' 2020', format='mixed').dt.month

    total_all_stars_games = (df['Opp'] != "@EAS")
    
    df = df[total_all_stars_games]
    df['Game_Type'] = df['Opp'].apply(lambda x: 'Away' if x.startswith('@') else 'Home')
    df['Min'] = df['Min'].apply(lambda x: int(x.split(':')[0]) + int(x.split(':')[1]) / 60)
    
    list_columns = df.columns

    values = st.multiselect("Select two variables:",list_columns,["Reb", "Pts"])

    # Creation of the line chart
    st.line_chart(df,x=values[0],y=values[1])

    # Creation of the bar chart 
    st.bar_chart(df,x=values[0],y=values[1])

    # Pairplot
    values_pairplot = st.multiselect("Select 4 variables:",list_columns,["Pts","Reb","Min","TO"])

    df2 = df[[values_pairplot[0],values_pairplot[1],values_pairplot[2],values_pairplot[3]]]
    pair = sns.pairplot(df2)
    st.pyplot(pair)


if app_page == 'Prediction':

    st.title("4. Prediction")
    
    st.write("Let's extract the Month played")
    df['Month'] = pd.to_datetime(df['Date'] + ' 2020', format='mixed').dt.month

    total_all_stars_games = (df['Opp'] != "@EAS")

    if total_all_stars_games.sum() != 0:
        st.write("Let's remove All Star Games since they're not a part of LeBron's season games")
    
    df = df[total_all_stars_games]
    st.write("Next, let's generate a new feature: is the Game Type - Away or Home")
    df['Game_Type'] = df['Opp'].apply(lambda x: 'Away' if x.startswith('@') else 'Home')
    
    
    st.write("Let's one hot encode it so that we can convert it to numerical columns for our prediction training model")
    game_dummies = pd.get_dummies(df['Game_Type'], prefix='', prefix_sep='')
    
    df = pd.concat([df, game_dummies], axis=1)
     # Convert to a numerical mins column
    st.write("Let's convert minutes from a mm:ss:SS format to numerical minutes out of 60")
    df['Min'] = df['Min'].apply(lambda x: int(x.split(':')[0]) + int(x.split(':')[1]) / 60)
    
    df = df.select_dtypes(exclude=['object'])

    st.dataframe(df.head(5))

    st.success("Now, we have all our numerical features ready to be used in our model")
    
    list_columns = df.columns.drop("Pts")
    input_lr = st.multiselect("Select variables:",list_columns,["FGA","OR","TO"])

    df2 = df[input_lr]

    # Step 1 splitting the dataset into X and y
    X= df2
    # target variable
    y= df["Pts"]

    # Step 2 splitting into 4 chuncks X_train X_test y_train y_test
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

    # Step3 Initialize the LinearRegression
    lr = LinearRegression()

    # Step4 Train model
    lr.fit(X_train,y_train)

    #Step5 Prediction 
    predictions = lr.predict(X_test)
    predictions_df = pd.Series(predictions, index=y_test.index, name="Predictions")

    result = pd.concat([predictions_df, y_test], axis=1)

    st.dataframe(result)

    #Stp6 Evaluation

    mae=metrics.mean_absolute_error(predictions,y_test)
    r2=metrics.r2_score(predictions,y_test)

    st.write("Mean Absolute Error:",mae)
    st.write("R2 output:",r2)

if app_page == 'Prediction':

    st.write("""While on the surface, projecting a player's points based on stats that are only available at the same time as points seems relatively useless it can give potent insights into player scouting, play time, and game strategy. The purpose of sports analytics is to find out which numbers ultimately impact winning, and using those numbers to create a winning team. By creating an algorithm that predicts Lebron’s points based on other stats from the present game a few things can be learned. One, if you are game planning against him, you are able to see which other areas correlate with his scoring lots of points. By limiting the Lakers two best scorers, Lebron James and Anthony Davis, you give yourselves a significantly better chance at winning against the Lakers. By looking into the stats you can see which stats correlate with a big scoring night from Lebron, and with a little more data research you can set a benchmark for how few points Lebron and Davis will need to score to give your team a greater than 50% chance of winning, and then decipher the ways in which you will need to limit Lebron’s other offensive statistics to make him reach this benchmark. 
But if you are on the Lakers staff you can use this to gameplan. Knowing the importance of scoring from a team’s best player, you can tailor your gameplan to allow Lebron to succeed in the areas that allow him to score the most points. Scoring more points is obviously how you win the game, so if we recognize the importance of scoring more points than we can understand the value in maximizing this. From the points prediction model and from the heatmap it is clear that Lebron James’ field goal percentage is more positively correlated than his 3 point percentage with the number of points he scores. Meaning that when drawing plays field goal percentage should be valued above three point percentage. Lebron James shot around 75% at the rim last year, while he shot roughly half of that from three. Furthermore his handle is better on his right side meaning that on the right side of the floor he has a better shot at getting to the rim while his shooting is generally better from the left. Now if we want to maximize the number of points he scores, the model tells us that we should prioritize his field goal percentage over his 3pt percentage and draw up plays sending him to his right and not his left. 
""")
    
