import streamlit as st

@st.cache_resource(experimental_allow_widgets=True)
def event_report():

    st.title('Game Report')
    match_cols = st.columns([0.27, 0.17, 0.1, 0.06, 0.1, 0.37])
    
    with match_cols[1]:
        st.title('Abol')
    with match_cols[2]:
        st.title('1')
        st.write('Number 8')
    with match_cols[3]:
        st.title(':')
    with match_cols[4]:
        st.title('1')
        st.write('Number 5')
    with match_cols[5]:
        st.title('Varnero')


    st.markdown(
    """
    <style>
        div[data-testid="column"]:nth-of-type(2)
        {
  
            text-align: end;
            
        } 

        div[data-testid="column"]:nth-of-type(3)
        {
            display: block;
            text-align: center;
            vertical-align: middle;
            padding: 0px;
            
  
        } 
    </style>
    """,unsafe_allow_html=True
)
    col1, col2, col3, col4, col5  = st.columns([0.35,0.1, 0.1, 0.1, 0.35])

    with col2:
        club1_evt = st.number_input('-', 0, 100, 0, label_visibility='collapsed', key='club1_evt')
        club1_fouls = st.number_input('-', 0, 100, 0, label_visibility='collapsed', key='club1_fouls')
        club1_yellow = st.number_input('-', 0, 100, 0, label_visibility='collapsed', key='club1_yellow')
        club1_corners = st.number_input('-', 0, 100, 0, label_visibility='collapsed', key='club1_corners')
        club1_throwins = st.number_input('-', 0, 100, 0, label_visibility='collapsed', key='club1_throwins')
        club1_attack = st.number_input('-', 0, 100, 0, label_visibility='collapsed', key='club1_attack')
        club1_goal = st.number_input('-', 0, 100, 0, label_visibility='collapsed', key='club1_goal')
        club1_possesion = st.number_input('-', 0, 100, 0, label_visibility='collapsed', key='club1_possesion')
    with col3:
        

        
        
        st.write('Off Target')
        st.write('Blocked')
        st.write('Free Kicks')
        st.write('Possession')
        st.write('Possession(HT)')
        st.write('Shots')
        st.write('Shots On Goal')
        st.write('Corner Kicks')
        st.write('Corner Kicks(HT)')
        st.write('Yellow Card')
        
        st.write('Pass')
        st.write('Pass Success')
        st.write('Foul')
        st.write('Heads')
        st.write('Head Success')
        st.write('Saves')
        st.write('Tackles')
        st.write('Dribbles')
        st.write('Throw ins')
        st.write('Shot on post')
        st.write('Tackle Success')
        st.write('Intercept')
        st.write('Assists')
        st.write('Attack')
        st.write('Dangerous attack')
    with col4:
        club2_evt = st.number_input('-', 0, 100, 0, label_visibility='collapsed', key='club2_evt')
        club2_fouls = st.number_input('-', 0, 100, 0, label_visibility='collapsed', key='club2_fouls')
        club2_yellow = st.number_input('-', 0, 100, 0, label_visibility='collapsed', key='club2_yellow')
        club2_corners = st.number_input('-', 0, 100, 0, label_visibility='collapsed', key='club2_corners')
        club2_throwins = st.number_input('-', 0, 100, 0, label_visibility='collapsed', key='club2_throwins')
        club2_attack = st.number_input('-', 0, 100, 0, label_visibility='collapsed', key='club2_attack')
        club2_goal = st.number_input('-', 0, 100, 0, label_visibility='collapsed', key='club2_goal')
        club2_possesion = st.number_input('-', 0, 100, 0, label_visibility='collapsed', key='club2_possesion')

