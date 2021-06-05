import numpy as np
import pandas as pd
def get_PUMA_dfs(data, PUMAlist, copy=False):
    ''' Converts a dataframe into multiple dataframes 
        as specified by the pumas
    
    Args:
        data: the original dataframe
        PUMAlist: list of pumas to be included in the dataframe
        copy: If true, function also returns copy of input dataframe
    Returns:
        df1: A copy of the original dataframe
        df_list: List of dataframes filtered by PUMAs
        
    '''
    df_list = []
    df1 = data.copy()
    for PUMAset in PUMAlist:
        df_list.append(df1.loc[df1['PUMA'].isin(PUMAset)])
    if copy:
        return df1,df_list
    return df_list

# Function that iteratively calculates the MOE with a primary weight column,
# replicate weight frame, and a prefix (for labeling) as the arguments
def moe_st_error(prim_weights, rep_weight_frame, p=''):
    ''' Calculates the moe and standard of error
    
    Args: 
        prim_weights: Primary weight column
        rep_weight_frame: Replicate weights
        p: Output column prefix
    Returns:
        prim_weights: Primary weight column
        SE_: Standard error
        MOE: Margin of Error
        lb: Lower bound
        ub: Upper bound
    '''
    z_score = 1.645  # 90% confidence
    # Subtracting the primary weights from each replicate weight then
    # squaring it and summing the squared differences.
    SE_ = np.sqrt(rep_weight_frame.apply(lambda x: (x - prim_weights) ** 2).sum(axis=1))
    MOE = 4/80 * 1.645 * SE_  # multiplying the SE by the z score to get MOE
    # +/- the MOE to and from the primary weights to get upper and lower bound
    lb, ub = prim_weights - MOE, prim_weights + MOE
    return pd.DataFrame({p+'Metric': prim_weights, p+' SE': SE_,
                         p+' MOE': MOE, p+' LB': lb, p+' UB': ub})

def recode(col_value):
    ''' Recodes values in the ESR column
    
    Args:
        col_value: one column value
    Returns 
        Unemp: unemployed
        Other: Employed
    '''
    if col_value == 3:
        return 'Unemp'
    elif col_value in [1, 2, 4, 5]:
        return 'Other'
    else:
        return np.NaN
    
    
def group(df,g1,g2,operation='sum'):
    p = 'PWGTP' # Primary weight
    w = list(df.filter(regex = r'PWGTP/d+?').columns) # Replicate weights
    
    if operation =='mean':
        age_df = age_dist = pd.DataFrame(df.groupby([g1,g2])
                        .mean())
    else:
        age_df = age_dist = pd.DataFrame(df.groupby([g1,g2])
                        .sum())
    
    calcs = moe_st_error(age_df[p], age_df[w]).reset_index()
    
    pivoted = pd.pivot_table(calcs[[g1,g2,'Metric']],index=g1,columns=g2)
    
    props = pivoted.apply(lambda x: x / pivoted.sum(axis=1))
    
    return pivoted, props

