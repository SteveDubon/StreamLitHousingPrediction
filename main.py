import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from math import sqrt

# ===========================================================================================================  LOAD DATA
@st.cache
def get_data(filename):
    housing_data = pd.read_csv(filename)
    return housing_data

housing_data = get_data('data/MLHousing_cleaned.csv')

# ==============================================================================================================  HEADER
st.title('Housing Price Prediction')
st.caption("Machine Learning Dashboard used to visualize current and future "
           "rental prices throughout the United States")
st.write('---')
st.subheader('Option 1: Pick Your Price Range and States')

# ============================================================================================================  FEATURES
st.sidebar.subheader('Option 2: Pick How You Want to Live')
st.sidebar.write('---')

params = {
    'housing_type': st.sidebar.selectbox('Type:', ('apartment', 'assisted living', 'condo', 'cottage/cabin', 'duplex',
                                                   'flat', 'house', 'in-law', 'land', 'loft', 'manufactured',
                                                   'townhouse')),
    'bedrooms': st.sidebar.slider('Number of Bedrooms:', 1, 8, 2),
    'bathrooms': st.sidebar.slider('Number of Bathrooms:', 0.0, 6.5, 2.0, 0.5),
    'cats': st.sidebar.radio('Cats Allowed:', ('yes', 'no')),
    'dogs': st.sidebar.radio('Dogs Allowed:', ('yes', 'no')),
    'smoking': st.sidebar.radio('Smoking Allowed:', ('yes', 'no')),
    'wheelchair_access': st.sidebar.radio('Wheelchair Access:', ('yes', 'no')),
    'vehicle_charge': st.sidebar.radio('Electric Vehicle Charger Access:', ('yes', 'no')),
    'furniture': st.sidebar.radio('Comes Furnished:', ('yes', 'no')),
    'laundry': st.sidebar.selectbox('Laundry Availability:', ('laundry in bldg', 'laundry on site', 'no laundry on site',
                                                              'w/d hookups', 'w/d in unit')),
    'parking': st.sidebar.selectbox('Parking Availability:', ('attached garage', 'carport', 'detached garage',
                                                              'no parking', 'off-street parking', 'street parking',
                                                              'valet parking'))
}

for key, val in params.items():
    if val == 'no' or val == 'apartment' or val == 'laundry in bldg' or val == 'attached garage':
        params[key] = 0
    elif val == 'yes' or val == 'assisted living' or val == 'laundry on site' or val == 'carport':
        params[key] = 1
    elif val == 'condo' or val == 'no laundry on site' or val == 'detached garage':
        params[key] = 2
    elif val == 'cottage/cabin' or val == 'w/d hookups' or val == 'no parking':
        params[key] = 3
    elif val == 'duplex' or val == 'w/d in unit' or val == 'off-street parking':
        params[key] = 4
    elif val == 'flat' or val == 'street parking':
        params[key] = 5
    elif val == 'house' or val == 'valet parking':
        params[key] = 6
    elif val == 'in-law':
        params[key] = 7
    elif val == 'land':
        params[key] = 8
    elif val == 'loft':
        params[key] = 9
    elif val == 'manufactured':
        params[key] = 10
    elif val == 'townhouse':
        params[key] = 11

# =============================================================================================================  MAPPING
states_tickers = ('AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN',
                  'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ',
                  'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA',
                  'WI', 'WV', 'WY')

col1, col2 = st.columns(2)
with col1:
    budget = st.slider('My rent budget ($ per month):',
                       500, 5000, (1500, 3000), 100)
with col2:
    state_selection = st.multiselect('Pick Your State', states_tickers)


budget_list = list(budget)
budget_first_element = budget_list[0]
budget_second_element = budget_list[1]
num_range = range(budget_first_element, budget_second_element)
num_list = list(num_range)

user_picks = housing_data.loc[(housing_data.type == params['housing_type']) &
                              (housing_data.beds == params['bedrooms']) &
                              (housing_data.baths == params['bathrooms']) &
                              (housing_data.cats_allowed == params['cats']) &
                              (housing_data.dogs_allowed == params['dogs']) &
                              (housing_data.smoking_allowed == params['smoking']) &
                              (housing_data.wheelchair_access == params['wheelchair_access']) &
                              (housing_data.electric_vehicle_charge == params['vehicle_charge']) &
                              (housing_data.comes_furnished == params['furniture']) &
                              housing_data.laundry_options.isin([params['laundry']]) &
                              housing_data.parking_options.isin([params['parking']]) &
                              housing_data.price.isin(num_list) &
                              housing_data.state.isin(state_selection)]

just_price = housing_data.loc[housing_data.price.isin(num_list)]
just_states = housing_data.loc[housing_data.state.isin(state_selection)]
price_state = housing_data.loc[housing_data.price.isin(num_list) &
                               housing_data.state.isin(state_selection)]

button = st.sidebar.button('Use My Picks')
map_this = ''
show_this = ''

if button == True:
    show_this = user_picks[['price', 'sqfeet', 'beds', 'baths', 'state']]
    map_this = user_picks[['latitude', 'longitude']]
    #print(user_picks)
else:
    if len(just_states) == 0:
        show_this = just_price[['price', 'sqfeet', 'beds', 'baths', 'state']]
        map_this = just_price[['latitude', 'longitude']]
    elif len(just_states) != 0:
        show_this = price_state[['price', 'sqfeet', 'beds', 'baths', 'state']]
        map_this = price_state[['latitude', 'longitude']]


st.map(map_this, 3)

with st.expander("See more details"):
    st.write("""
        The table below shows all of the results based on the user's inputs: Price, Square Footage, Bedrooms, Bathrooms
        and States.
    """)
    st.write(show_this)

st.write('---')
st.subheader('Option 3: Use Machine Learning Model to Predict the Price')
st.caption("Machine Learning Dashboard used to visualize current and future "
           "rental prices throughout the United States")
st.write('---')

# ====================================================================================================  MACHINE LEARNING
test_size = st.slider('Pick Test Size', 0.05, 0.5, 0.15, step=0.05)

@st.cache
def get_models():
    y = housing_data['price']
    X = housing_data[['type', 'beds', 'baths', 'cats_allowed', 'dogs_allowed', 'smoking_allowed', 'wheelchair_access',
                      'electric_vehicle_charge', 'comes_furnished', 'laundry_options', 'parking_options']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True)
    models = [DummyRegressor(strategy='mean'),
              RandomForestRegressor(n_estimators=170, max_depth=25),
              DecisionTreeRegressor(max_depth=25),
              GradientBoostingRegressor(learning_rate=0.01, n_estimators=200, max_depth=5),
              LinearRegression(n_jobs=10, normalize=True)]
    df_models = pd.DataFrame()

    temp = {}
    print(X_test)

    for model in models:
        print(model)
        m = str(model)
        temp['Model'] = m[:m.index('(')]
        model.fit(X_train, y_train)
        temp['RMSE_Price'] = sqrt(mse(y_test, model.predict(X_test)))
        temp['Pred Value'] = model.predict(pd.DataFrame(params, index=[0]))[0]
        print('RMSE score', temp['RMSE_Price'])
        df_models = df_models.append([temp])
    df_models.set_index('Model', inplace=True)
    pred_value = df_models['Pred Value'].iloc[[df_models['RMSE_Price'].argmin()]].values.astype(float)
    return pred_value, df_models

def run_data():
    df_models = get_models()[0][0]
    st.write('Given your parameters, the predicted value is **${:.2f}**'.format(df_models))

btn = st.button("Predict")
if btn:
    run_data()
else:
    pass

with st.expander("See more details"):
    st.subheader('Additional Information')

    if st.checkbox('Show ML Models'):
        run_data()
        df_models = get_models()[1]
        #df_models
        st.write('**This diagram shows root mean sq error for all models**')
        st.bar_chart(df_models['RMSE_Price'])
