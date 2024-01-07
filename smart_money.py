import numpy as np
import pandas as pd

from enums import Direction, Channel
from scipy.ndimage import  maximum_filter1d, minimum_filter1d
from scipy.signal import find_peaks
from scipy import stats

class SmartMoney:

    filter_size: int    = 7
    has_errors: bool    = False

    
    def is_uptrend(self, df: pd.DataFrame) -> bool:
        if self.meets_requirement(df=df) == False:
            return False
        return (
                    df.loc[df['is_resistance'] == 1, 'high'].iloc[-1] > df.loc[df['is_resistance'] == 1, 'high'].iloc[-2] and 
                    df.loc[df['is_support'] == 1, 'low'].iloc[-1] > df.loc[df['is_support'] == 1, 'low'].iloc[-2]  
                )

    def is_downtrend(self, df: pd.DataFrame) -> bool:
        if self.meets_requirement(df=df) == False:
            return False
        return (
                    df.loc[df['is_resistance'] == 1, 'high'].iloc[-1] < df.loc[df['is_resistance'] == 1, 'high'].iloc[-2] and 
                    df.loc[df['is_support'] == 1, 'low'].iloc[-1] < df.loc[df['is_support'] == 1, 'low'].iloc[-2]  
                )
    
    def is_in_impulse_phase(self, df:pd.DataFrame, is_uptrend = True) -> bool:
        if self.meets_requirement(df=df, minimum_required=2) is False:
            return False

        if is_uptrend:
            return df[df['is_resistance'] == 1].index[-1] < df[df['is_support'] == 1].index[-1]
        else:
            return df[df['is_resistance'] == 1].index[-1] > df[df['is_support'] == 1].index[-1]
    
    def is_in_pullback_phase(self, df: pd.DataFrame, candle: pd.Series = None) -> bool:

        if self.meets_requirement(df=df, minimum_required=2) is False:
            return False

        candle = df.iloc[-1] if candle == None else candle
        return all([
                    df.loc[df['is_resistance'] == 1, 'high'].iloc[-1] > candle.high,
                    df.loc[df['is_support'] == 1, 'low'].iloc[-1] < candle.low
        ])

    def is_in_breakout_phase(self, df: pd.DataFrame, is_uptrend: bool = False, candle: pd.Series = None) -> bool:
        if self.meets_requirement(df=df, minimum_required=2) is False:
            return False

        candle = df.iloc[-1] if candle == None else candle
        if is_uptrend:
            return candle.high > df.loc[df['is_resistance'] == 1, 'high'].iloc[-1]
        else:
            return candle.low < df.loc[df['is_support'] == 1, 'low'].iloc[-1]

    # Check if price broke above/below minima/maxima and is now coming for a retest
    def is_retesting(self, df: pd.DataFrame, direction: Direction) -> bool:
        if self.meets_requirement(df=df, minimum_required=2) is False:
            return False

        if direction == Direction.UP:

            # is in pullback phase
            highest_point = df.loc[df['is_resistance'] == 1, 'high'].iloc[-1]
            break_out_leg = highest_point - df.loc[df['is_resistance'] == 1, 'high'].iloc[-2]
            swing_leg = df.loc[df['is_resistance'] == 1, 'high'].iloc[-2]  - df.loc[df['is_support'] == 1, 'low'].iloc[-1]

            # is is break out phase
            if self.is_in_breakout_phase(df=df, is_uptrend=True):
                highest_point = df.loc[df['is_resistance'] == 1, 'high'].iloc[-1:].max()

                break_out_leg = highest_point - df.loc[df['is_resistance'] == 1, 'high'].iloc[-1]
                swing_leg = df.loc[df['is_resistance'] == 1, 'high'].iloc[-1]  - df.loc[df['is_support'] == 1, 'low'].iloc[-1]

                if (break_out_leg / swing_leg * 100) < 20:
                    return False


            pull_back_leg = highest_point - df.iloc[-1].low
            if (pull_back_leg / break_out_leg) * 100 >= 90:
                return True

        elif direction == Direction.DOWN:
            # is in pullback phase
            lowest_point = df.loc[df['is_support'] == 1, 'low'].iloc[-1]
            break_out_leg = df.loc[df['is_support'] == 1, 'low'].iloc[-2] - lowest_point
            swing_leg = df.loc[df['is_resistance'] == 1, 'high'].iloc[-1]  - df.loc[df['is_support'] == 1, 'low'].iloc[-2]

            # is is break out phase
            if self.is_in_breakout_phase(df=df, is_uptrend=True):
                lowest_point = df.loc[df['is_support'] == 1, 'low'].iloc[-1:].min()

                break_out_leg = df.loc[df['is_support'] == 1, 'low'].iloc[-1] - lowest_point
                swing_leg = df.loc[df['is_resistance'] == 1, 'high'].iloc[-1]  - df.loc[df['is_support'] == 1, 'low'].iloc[-1]

                if (break_out_leg / swing_leg * 100) < 20:
                    return False


            pull_back_leg = df.iloc[-1].high - lowest_point
            if (pull_back_leg / break_out_leg) * 100 >= 80:
                return True

        
        return False

            
    def get_left_and_right(self, df: pd.DataFrame, divide_by_high = True) -> tuple[pd.DataFrame, pd.DataFrame]:
       
       # Get the lowest/highest support df
        off_set = df['low'].idxmin() if divide_by_high == False else df['high'].idxmax()

        # Get list of df before lowest support
        left    =   df[:off_set]

        # take only resistance and leave out support
        # left    =   left[left['is_resistance'] == 1]
        left.reset_index(drop=True, inplace=True) 


        # Get list aft the df after loweset support
        right   =   df[off_set:]

        # take only resistance and leave out support
        # right   =   right[right['is_resistance'] == 1]
        right.reset_index(drop=True, inplace=True)

        return pd.DataFrame(left), pd.DataFrame(right)


    def has_bull_choch(self, df: pd.DataFrame, in_pullback_phase = False, with_first_impulse = False) -> bool:
        if df[df['is_resistance'] == 1].empty:
            return False
        
        left, right = self.get_left_and_right(df = df, divide_by_high=False)

        if len(left[left['is_resistance'] == 1]) < 1 or right.shape[0] < 1:
            return False

        # if we only want CHoCH that broke on first impulse move
        if with_first_impulse:
            if left.loc[left['is_resistance'] == 1, 'high'].iloc[-1] > right.loc[right['is_resistance'] == 1, 'high'].iloc[0] :
                return False

        # if we want CHoCH in pullback phase
        if in_pullback_phase:
            if right.iloc[right[right['is_resistance'] == 1].index[-1], right.columns.get_loc('high')] < right['high'].iloc[-1]:
                return False
        
        tmp = right[right['high'] > left.loc[left['is_resistance'] == 1, 'high'].iloc[-1]]
        if tmp.shape[0] > 0 :
            return True
        return False


    def has_bear_choch(self, df: pd.DataFrame, in_pullback_phase = False, with_first_impulse = False) -> bool:
        if df[df['is_support'] == 1].empty:
            return False
        
        left, right = self.get_left_and_right(df = df, divide_by_high=True)

        if len(left[left['is_support'] == 1]) < 1 or right.shape[0] < 1:
            return False

        if with_first_impulse:
            if  right.loc[right['is_support'] == 1, 'low'].iloc[-1] > left.loc[left['is_support'], 'low'].iloc[-1]:
                return False

        if in_pullback_phase:
            if right.loc[right['is_support'] == 1, 'low'].iloc[-1] > right.low.iloc[-1] :
                return False
        
        tmp = right[right['low'] < df.loc[df['is_support'] == 1, 'low'].iloc[-1]]
        if tmp.shape[0] > 0 :
            return True
        return False
    
    def get_impulse_and_pullback_value(self, df: pd.DataFrame, is_uptrend: bool, on_last_candle: bool = False) -> tuple[float, float]:

        impulse_leg = 1
        pullback = 2

        
        if self.is_uptrend(df=df):

            if self.is_in_impulse_phase(df=df, is_uptrend=is_uptrend) is True:
                impulse_leg = df.loc[df['is_resistance'] == 1, 'high'].iloc[-1] - df.loc[df['is_support'] == 1, 'low'].iloc[-2]
            else:
                impulse_leg = df.loc[df['is_resistance'] == 1, 'high'].iloc[-1] - df.loc[df['is_support'] == 1, 'low'].iloc[-1]


            if on_last_candle is True:

                pullback =  df.loc[df['is_resistance'] == 1, 'high'].iloc[-1] - df.low.iloc[-1] 
            else:
                pullback =  df.loc[df['is_resistance'] == 1, 'high'].iloc[-1] - df.loc[df[df['is_resistance'] == 1].index[-1]:]['low'].min()

        elif self.is_downtrend(df=df):
            
            if self.is_in_impulse_phase(df=df, is_uptrend=is_uptrend) is True:
                impulse_leg = df.loc[df['is_resistance'] == 1, 'high'].iloc[-2] - df.loc[df['is_support'] == 1, 'low'].iloc[-1]
            else:
                impulse_leg = df.loc[df['is_resistance'] == 1, 'high'].iloc[-1] - df.loc[df['is_support'] == 1, 'low'].iloc[-1]
                
            if on_last_candle is True:
                pullback = df.high.iloc[-1] - df.loc[df['is_support'] == 1, 'low'].iloc[-1] 
            else:
                pullback = df.loc[df[df['is_support'] == 1].index[-1]:]['high'].max() - df.loc[df['is_support'] == 1, 'low'].iloc[-1]

        else:

            if self.meets_requirement(df=df, minimum_required=1) is not True:
                 return impulse_leg, pullback

            impulse_leg = df.loc[df['is_resistance'] == 1, 'high'].iloc[-1] - df.loc[df['is_support'] == 1, 'low'].iloc[-1]

            if df.loc[df['is_resistance'] == 1].index[-1] > df.loc[df['is_support'] == 1].index[-1]:

                if on_last_candle is True:
                    pullback =  df.loc[df['is_resistance'] == 1, 'high'].iloc[-1] - df.low.iloc[-1] 
                else:
                    pullback =  df.loc[df['is_resistance'] == 1, 'high'].iloc[-1] - df.loc[df[df['is_resistance'] == 1].index[-1]:]['low'].min()
            else:
                if on_last_candle is True:
                    pullback = df.high.iloc[-1] - df.loc[df['is_support'] == 1, 'low'].iloc[-1] 
                else:
                    pullback = df.loc[df[df['is_support'] == 1].index[-1]:]['high'].max() - df.loc[df['is_support'] == 1, 'low'].iloc[-1]

        return impulse_leg, pullback
    
    def is_in_discount_zone(self, df: pd.DataFrame, is_uptrend: bool, on_last_candle: bool = False) -> bool:
        leg, pullback  = self.get_impulse_and_pullback_value(df=df, is_uptrend=is_uptrend, on_last_candle=on_last_candle)
        pullback_perc = pullback / leg * 100
        return all([pullback_perc >= 50, pullback_perc <=100])

    def meets_requirement(self, df: pd.DataFrame, minimum_required: int = 2) -> bool:
        return len(df.loc[df['is_resistance'] == 1]) >= minimum_required and len(df.loc[df['is_support'] == 1]) >= minimum_required

    # Check if the latest minimas and maximas form a channel
    def calculate_channel_params(self, row, df) -> tuple[float, float]:

        highs = df.loc[df['is_resistance'] == 1, 'high'].values[-3:]
        idxhighs = df.loc[df['is_resistance'] == 1].index.values[-3:]
        lows = df.loc[df['is_support'] == 1, 'low'].values[-3:]
        idxlows = df.loc[df['is_support'] == 1].index.values[-3:]

        total_length = len(lows) + len(highs)

        if len(lows) >= 2 and len(highs) >= 2 and total_length >= 5:
            # Calculate linear regression for lows and highs
            sl_lows, interc_lows, r_value_l, _, _ = stats.linregress(idxlows, lows)
            sl_highs, interc_highs, r_value_h, _, _ = stats.linregress(idxhighs, highs)


            r_sq_h = r_value_h**2
            r_sq_l = r_value_l**2

            if sl_highs and sl_lows and (r_sq_l >= 0.9 <= r_sq_h):
                channel = Channel(
                                     is_channel         = True,
                                     lower_slop         = sl_lows,
                                     upper_slop         = sl_highs,
                                     lower_intersect    = interc_lows,
                                     upper_intersect    = interc_highs,
                                     lower_percent      = r_sq_l,
                                     upper_percent      = r_sq_h,
                                     lower_line_val     = (sl_lows*df.index + interc_lows),
                                     upper_line_val     = (sl_highs*df.index + interc_highs)
                             )



                return self._channel_buy_zone(df=df, channel=channel), self._channel_sell_zone(df=df, channel=channel)

        return 0, 0

    # Just checking is the price is getting closer to the upper part of the channel
    # it return the percentage
    def _channel_sell_zone(self, df: pd.DataFrame, channel: Channel) -> float:
        if not channel.is_channel:
            return 0.0

        lower_line_val = channel.lower_slop * df.index[-1] + channel.lower_intersect
        upper_line_val = channel.upper_slop * df.index[-1] + channel.upper_intersect

        return (df['high'].iloc[-1] - lower_line_val) / (upper_line_val - lower_line_val) * 100


    # Just checking if the price is getting closer to the lower part of the channel
    # return percentage
    def _channel_buy_zone(self, df: pd.DataFrame, channel: Channel) -> float:
        if not channel.is_channel:
            return 0.0

        lower_line_val = channel.lower_slop * df.index[-1] + channel.lower_intersect
        upper_line_val = channel.upper_slop * df.index[-1] + channel.upper_intersect

        return (upper_line_val - df['low'].iloc[-1]) / (upper_line_val - lower_line_val) * 100


    # This is when one candle is marked as a minima and maxima at the same time.
    def remove_candles_with_minimas_and_maximas(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        minimas = df.index[df['is_support'] == 1].to_numpy()
        maximas = df.index[df['is_resistance'] == 1].to_numpy()

        if len(maximas) < 1 and len(minimas) < 1:
            return minimas, maximas

        unwanted = df.index[(df['is_resistance'] == 1) & (df['is_support'] == 1)].to_numpy()

        for i in range(0, unwanted.shape[0]):
            index = unwanted[i]
            minimas_condition = np.argwhere(np.isin(minimas, index))
            maximas_condition = np.argwhere(np.isin(maximas, index))

            if len(unwanted) > 2:
                if df['low'].iloc[index] < df['low'].iloc[index-1]:
                    maximas = np.delete(maximas, maximas_condition)
                else:
                    minimas = np.delete(minimas, minimas_condition)
            else:
                maximas = np.delete(maximas, maximas_condition)
                minimas = np.delete(minimas, minimas_condition)

        return minimas, maximas


    # This is when we have two minimas/maximas next to each other,
    # As per SMC this cannot happen, we should have: minima, maxima, minima, maxima etc
    def remove_crowded_maximas_and_minimas(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        _minimas = df.index[df['is_support'] == 1].to_numpy()
        _maximas = df.index[df['is_resistance'] == 1].to_numpy()

        minimas = _minimas
        maximas = _maximas


        # Before anything, lets clean crowded resistances before first support
        if len(minimas) > 0:
            new_df = df.loc[:minimas[0]]
            new_resistances = new_df.index[new_df['is_resistance'] == 1].to_numpy()
            if len(new_resistances) > 1:
                highest = new_df.high[new_resistances].idxmax()
                others = new_resistances[new_resistances != highest]
                condition = np.argwhere(np.isin(maximas, others))
                maximas = np.delete(maximas, condition)

        for minima in range(1,_minimas.shape[0]):
            prev_minima = _minimas[minima -1] # previous minima
            cur_minima = _minimas[minima] # current minima
            
            new_maximas = maximas[np.logical_and(prev_minima < maximas, cur_minima > maximas)] 

            if new_maximas.size == 0 : # If there's no maxima between two minimas

                if df.low[prev_minima] < df.low[cur_minima]: # If the previous minima was lower than the current minimas
                    highest_id = np.where(minimas == cur_minima)
                    minimas = np.delete(minimas, highest_id)
                    _minimas[minima] = prev_minima
                else:
                    highest_id = np.where(minimas == prev_minima)
                    minimas = np.delete(minimas, highest_id)
            elif new_maximas.size > 1 : # if results if greater than 1 we take the highest and remove the rest

                # We keep the highest
                highest = df.high[new_maximas].idxmax() # highest maxima
                others = new_maximas[new_maximas != highest] # Other unwanted maximas
                condition = np.argwhere(np.isin(maximas, others)) # Get indices for unwanted minimas
                maximas = np.delete(maximas, condition)


        # Now we delete crowded maximas after the last minima
        last_support_indices = df.index[df['is_support'] == 1].to_list()
        if len(last_support_indices) > 0:
            new_df = df.loc[last_support_indices[-1]:]
            new_resistances = new_df.index[new_df['is_resistance'] == 1].to_numpy()
            if len(new_resistances) > 0:
                highest = new_df.high[new_resistances].idxmax()
                if len(new_resistances) > 1:
                    others = new_resistances[new_resistances != highest]
                    condition = np.argwhere(np.isin(maximas, others))
                    maximas = np.delete(maximas, condition)

                # Now if price went higher than our last maxima, it's not valid anymore
                after_high_df = new_df.loc[highest:]

                any_greater = after_high_df.loc[after_high_df.high > new_df.high[highest]]
                if len(any_greater) > 0:
                    highest_index = np.where(maximas == highest)
                    maximas = np.delete(maximas, highest_index)


            elif len(new_resistances) == 0:
                #And if the price went lower than our last minima, then the minima is not valid anymore
                last_minima_index = df.index[df['is_support'] == 1].to_list()[-1]
                new_df = df.loc[last_minima_index:]

                any_lower = new_df.loc[new_df.low < df.low[last_minima_index]] 
                if len(any_lower) > 0:
                    last_minima_index = np.where(minimas == last_minima_index )
                    minimas = np.delete(minimas, last_minima_index)


        return minimas, maximas
    
    # This is when we have two minimas/maximas next to each other,
    # As per SMC this cannot happen, we should have: minima, maxima, minima, maxima etc
    def has_crowded_maximas_or_minimas(self, df: pd.DataFrame) -> bool:
        _minimas = df.index[df['is_support'] == 1].to_numpy()
        _maximas = df.index[df['is_resistance'] == 1].to_numpy()

        maximas = _maximas

        hasCrowdedMaximasBool = False

        # Before anything, lets clean crowded resistances before first support
        first_support_indices = df.index[df['is_support'] == 1].to_list()
        if len(first_support_indices) > 0:
            new_df = df.loc[:first_support_indices[0]]

            new_resistances = new_df.index[new_df['is_resistance'] == 1].to_list()
            if len(new_resistances) > 1:
                hasCrowdedMaximasBool = True
                
      

        for minima in range(1,_minimas.shape[0]):
            prev_minima = _minimas[minima -1] # previous minima
            cur_minima = _minimas[minima] # current minima
            
            new_maximas = maximas[np.logical_and(prev_minima < maximas, cur_minima > maximas)] # Get maximas between current minima and previous minima

            if new_maximas.size == 0 : # If there's no maxima between two minimas
                hasCrowdedMaximasBool = True
                break
            elif new_maximas.size > 1 : # if results if greater than 1 we take the highest and remove the rest
                hasCrowdedMaximasBool = True
                break

        # Now we delete crowded maximas after the last minima
        last_support_indices = df.index[df['is_support'] == 1].to_list()
        if len(last_support_indices) > 0:
            new_df = df.loc[last_support_indices[-1]:]
            new_resistances = new_df.index[new_df['is_resistance'] == 1].to_list()
            if len(new_resistances) > 0:
                highest = new_df.high[new_resistances].idxmax()
                if len(new_resistances) > 1:
                    hasCrowdedMaximasBool = True

                # Now if price went higher than our last maxima, it's not valid anymore
                after_high_df = new_df.loc[highest:]

                any_greater = after_high_df.loc[after_high_df.high > new_df.high[highest]]
                if len(any_greater) > 0:
                    hasCrowdedMaximasBool = True


            elif len(new_resistances) == 0:
                #And if the price went lower than our last minima, then the minima is not valid anymore
                last_minima_index = df.index[df['is_support'] == 1].to_list()[-1]
                new_df = df.loc[last_minima_index:]

                any_lower = new_df.loc[new_df.low < df.low[last_minima_index]] 
                if len(any_lower) > 0:
                    hasCrowdedMaximasBool = True


        return hasCrowdedMaximasBool
            

    def _remove_weak_minima_and_maxima(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        minimas = df.index[df['is_support'] == 1].to_numpy()
        maximas = df.index[df['is_resistance'] == 1].to_numpy()

        curr_minima  = minimas[-1]
        prev_minima  = minimas[-2]

        curr_maxima = maximas[-1]

        if df.low[curr_minima] >= df.low[prev_minima] and df.high[maximas[-1]] <= df.high[maximas[-2]]:

            if curr_maxima < curr_minima:
                index = maximas[-1] 
                df_upto_now = df.loc[:index]
            else:
                df_upto_now = df.loc[:]

            if self.is_uptrend(df=df_upto_now):

                if curr_maxima > curr_minima:
                    minimas = np.delete(minimas, np.argwhere(minimas == minimas[-1])[0])
                else:
                    maximas = np.delete(maximas, np.argwhere(maximas == maximas[-1])[0])

            elif self.is_downtrend(df=df_upto_now):
                if curr_maxima < curr_minima:
                    minimas = np.delete(minimas, np.argwhere(minimas == minimas[-1])[0])
                else:
                    maximas = np.delete(maximas, np.argwhere(maximas == maximas[-1])[0])
                
            else:

                if curr_maxima < curr_minima:
                    minimas = np.delete(minimas, np.argwhere(minimas == minimas[-1])[0])
                else:
                    maximas = np.delete(maximas, np.argwhere(maximas == maximas[-1])[0])

        return minimas, maximas
       

    # This is when the latest minima (-1) is greater than the previous minima (-2) 
    # AND latest maxima (-1) is lower than previous maxima (-2)
    # As per Smart Money concept this cannot happen, price has to make higher highs and higher lows or Lower Lows and Lower highs
    def remove_weak_minimas_and_maximas(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        _minimas = df.index[df['is_support'] == 1].to_numpy()
        _maximas = df.index[df['is_resistance'] == 1].to_numpy()

        minimas = _minimas
        maximas = _maximas

        for minima in range(1, _minimas.shape[0]):
            curr_minima  = _minimas[minima]
            
            df_upto_now = df.loc[:curr_minima + 1] # get all candles up to now, plus 1 make current candle inclusive
            lastest_maximas = df_upto_now.index[df_upto_now['is_resistance'] == 1].to_numpy()
            lastest_minimas = df_upto_now.index[df_upto_now['is_support'] == 1].to_numpy()

            if len(lastest_maximas) < 2 or 2 > len(lastest_minimas):
                continue

            minimas, maximas = self._remove_weak_minima_and_maxima(df=df_upto_now)
            df.loc[:curr_minima + 1, 'is_resistance'] = 0
            df.loc[:curr_minima + 1, 'is_support'] = 0

            if len(maximas) > 0:
                df.loc[maximas, 'is_resistance'] = 1

            if len(minimas) > 0:
                df.loc[minimas, 'is_support'] = 1        
           
        # Now we check for maxima after the last minima

        if len(maximas) < 2 or 2 > len(minimas):
            return minimas, maximas 
        
        return  self._remove_weak_minima_and_maxima(df=df)
    
    # This is when the latest minima (-1) is greater than the previous minima (-2) 
    # AND latest maxima (-1) is lower than previous maxima (-2)
    # As per Smart Money concept this cannot happen, price has to make higher highs and higher lows or Lower Lows and Lower highs
    def has_weak_minimas_or_maximas(self, df: pd.DataFrame) -> bool:
        _minimas = df.index[df['is_support'] == 1].to_numpy()

        has_weak_minima_or_maxima = False

        for minima in range(1, _minimas.shape[0]):
            curr_minima  = _minimas[minima]
            prev_minima  = _minimas[minima-1]
                    
            df_upto_now = df.loc[:curr_minima + 1] # get all candles up to now, plus 1 make current candle inclusive
            lastest_maximas = df_upto_now.index[df_upto_now['is_resistance'] == 1].to_numpy()
            lastest_minimas = df_upto_now.index[df_upto_now['is_support'] == 1].to_numpy()

            if len(lastest_maximas) < 2 or 2 > len(lastest_minimas):
                continue

            if df.low[curr_minima] >= df.low[prev_minima] and df.high[lastest_maximas[-1]] <= df.high[lastest_maximas[-2]]:
                has_weak_minima_or_maxima = True


        # Now we check for maxima after the last minima
        
        lastest_maximas = df.index[df['is_resistance'] == 1].to_numpy()
        lastest_minimas = df.index[df['is_support'] == 1].to_numpy()

        if len(lastest_maximas) < 2 or 2 > len(lastest_minimas):
            return has_weak_minima_or_maxima 
        

        curr_minima  = _minimas[-1]
        prev_minima  = _minimas[-2]

        if df.low[curr_minima] >= df.low[prev_minima] and df.high[lastest_maximas[-1]] <= df.high[lastest_maximas[-2]]:
            has_weak_minima_or_maxima = True

                
        return has_weak_minima_or_maxima
    
    def clean_minimas_and_maximas(self, df: pd.DataFrame, strict = False) -> pd.DataFrame:

        # remove candle that has minima and maxima at the same time
        minimas, maximas = self.remove_candles_with_minimas_and_maximas(df=df)
        df.loc[:, 'is_resistance'] = 0
        df.loc[:, 'is_support'] = 0

        df.loc[maximas, 'is_resistance'] = 1        
        df.loc[minimas, 'is_support'] = 1        

        # Remove crowded minimas
        minimas, maximas = self.remove_crowded_maximas_and_minimas(df=df)
        df.loc[:, 'is_resistance'] = 0
        df.loc[:, 'is_support'] = 0

        df.loc[maximas, 'is_resistance'] = 1        
        df.loc[minimas, 'is_support'] = 1        

        self.has_errors = True if self.has_crowded_maximas_or_minimas(df=df) else False
        
        if strict and self.has_errors == False:
            minimas, maximas = self.remove_weak_minimas_and_maximas(df=df)
            df.loc[:, 'is_resistance'] = 0
            df.loc[:, 'is_support'] = 0

            df.loc[maximas, 'is_resistance'] = 1
            df.loc[minimas, 'is_support'] = 1        
                 
            has_weeak_mins = self.has_weak_minimas_or_maximas(df=df)
            has_crowded = self.has_crowded_maximas_or_minimas(df=df)

            self.has_errors = True if has_crowded or has_weeak_mins else False


        return df

    def look_back(self, df: pd.DataFrame) -> int:
        return round(np.mean(df['high'] - df['low']))        

    # Get support zones
    def _get_supports(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 1:
            return df

        smoothed_low = minimum_filter1d(df.low, self.filter_size) if self.filter_size > 0 else df.low
        minimas, _ = find_peaks(x=-smoothed_low, prominence=self.look_back(df=df))

        if len(minimas) > 0:
            df.loc[minimas, 'is_support'] = 1        
        return df

    # Get resistances zones
    def _get_resistances(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 1:
            return df

        smoothed_high = maximum_filter1d(df.high, self.filter_size) if self.filter_size > 0 else df.high
        maximas, _ = find_peaks(smoothed_high, prominence=self.look_back(df=df))
        if len(maximas) > 0:
            df.loc[maximas, 'is_resistance'] = 1
        return df

    def get_supports_and_resistances(self, df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
        df['is_support'] = 0
        df['is_resistance'] = 0
        df = self._get_resistances(df=df)
        df = self._get_supports(df=df)

        df = self.clean_minimas_and_maximas(df=df, strict=strict)

        if self.meets_requirement(df=df, minimum_required=1) == False:
            return df

        while self.has_errors:
            df = self.clean_minimas_and_maximas(df=df, strict=strict)
   
        return df


    
