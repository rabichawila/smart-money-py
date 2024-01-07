The Documentation

# get_supports_and_resistances():

This function simply take a pandas Dataframe and calculate minimas and maximas (support and resistance), it then adds two colums ( `is_support` and `is_resistance`).

# get_supports()

Uses scipy find_peaks function to calculate minimas/support, it is called in `get_supports_and_resistances` function.

# get_resistances()

Uses scipy find_peaks function to calculate maximas/resistances, it is called in `get_resistances` function.

# is_uptrend()

Uses two most recent maximas/resistances to determine if we're in an uptrend or not as per the definition of uptrend in Smart Money Concept.

# is_downtrend()

Uses two most recent minimas/maximas to determine if we're in a downtrend or not as per the definition of downtrend in Smart Money Concept.






