"""
script for adding attributes to TEA variables
"""

import re

equal_vars = {'EM': 'tEX'}


def get_attrs(vname=None, dec=False, spread=None, period='', data_unit=''):
    """
    get attributes for TEA variables
    Args:
        vname: variable name
        dec: decadal mean added to variable name
        spread: None, 'upper' or 'lower'. Set if spread estimator
        period: climatic time period
        data_unit: data unit (e.g. 'K', 'degC', 'mm')

    Returns:
        attributes: dict with attributes
    """

    if 'DTEA' in vname and not data_unit:
        data_unit = '100 km^2'
    attrs = {
        'ctp': {'long_name': f'climatic time period ({period})'},
        'CTP': {'long_name': f'start date of climatic time period {period}'},
        'decadal': {
            'long_name': f'center year of decadal indicators for climatic time period {period}', },
        'amplification': {
            'long_name': 'center year of decadal amplification factors for climatic time period'
                         f' {period}'},

        'DTEC': {'long_name': 'daily threshold exceedance count', 'units': '1'},
        'DTED': {'long_name': 'daily threshold exceedance duration', 'units': 'h'},
        'DTEM': {'long_name': 'daily threshold exceedance magnitude', 'units': data_unit},
        'DTEA': {'long_name': 'daily threshold exceedance area', 'units': data_unit},
        'DTEMA': {'long_name': 'daily threshold exceedance magnitude * area (auxiliary)',
                  'units': f'100 km^2 {data_unit}'},
        'DTEM_Max': {'long_name': 'daily maximum grid cell exceedance magnitude',
                     'units': data_unit},
        f'DTEEC': {'long_name': f'daily threshold exceedance event count', 'units': '1'},

        'EF': {'long_name': 'event frequency', 'units': 'yr^-1', 'metric_type': 'basic'},
        'doy_first': {'long_name': 'day of first event occurrence', 'units': '1',
                      'metric_type': 'basic'},
        'doy_last': {'long_name': 'day of last event occurrence', 'units': '1',
                     'metric_type': 'basic'},
        'AEP': {'long_name': 'annual exposure period', 'units': 'months',
                'metric_type': 'basic'},
        'ED': {'long_name': 'cumulative events duration', 'units': f'd yr^-1',
               'metric_type': 'compound'},
        'ED_avg': {'long_name': 'average events duration', 'units': 'd',
                   'metric_type': 'basic'},
        'EM': {'long_name': 'cumulative exceedance magnitude', 'units': f'{data_unit} d yr^-1',
               'description': 'expresses the temporal events extremity (tEX)',
               'metric_type': 'compound'},
        'EM_avg': {'long_name': 'average exceedance magnitude', 'units': data_unit,
                   'metric_type': 'basic'},
        'threshold_avg': {'long_name': 'average threshold value', 'units': data_unit,
                          'metric_type': 'basic'},
        'EM_avg_Md': {'long_name': 'average daily-median exceedance magnitude',
                      'units': data_unit, 'metric_type': 'basic'},
        'EM_Md': {'long_name': 'cumulative daily-median exceedance magnitude',
                  'units': f'{data_unit} d yr^-1', 'metric_type': 'compound'},
        'EM_Max': {'long_name': 'cumulative maximum exceedance magnitude',
                   'units': f'{data_unit} d yr^-1', 'metric_type': 'compound'},
        'EM_avg_Max': {'long_name': 'average maximum exceedance magnitude',
                       'units': data_unit, 'metric_type': 'basic'},
        'EA_avg': {'long_name': 'average exceedance area', 'units': '100 km^2',
                   'metric_type': 'basic'},
        'DM': {'long_name': 'duration-magnitude indicator', 'units': f'{data_unit} d',
               'metric_type':
                   'compound'},
        'TEX': {'long_name': 'total events extremity',
                'units': f'100 km^2 {data_unit} d yr^-1', 'metric_type':
                    'compound'},
        'hTEX': {'long_name': 'hourly total events extremity',
                 'units': f'100 km^2 {data_unit} h yr^-1',
                 'metric_type': 'compound'},
        'ES_avg': {'long_name': 'average event severity',
                   'units': f'100 km^2 {data_unit} d', 'metric_type': 'compound'},
        'hES_avg': {'long_name': 'average hourly event severity',
                    'units': f'100 km^2 {data_unit} h', 'metric_type': 'compound'},
        'tEX': {'long_name': 'temporal events extremity', 'units': f'{data_unit} d yr^-1',
                'metric_type':
                    'compound'},
        'htEX': {'long_name': 'hourly temporal events extremity',
                 'units': f'{data_unit} h yr^-1', 'metric_type':
                     'compound'},
        'H_AEHC_avg': {'long_name': 'average daily atmospheric boundary layer exceedance '
                                    'heat content', 'units': 'PJ d^-1',
                       'metric_type': 'compound'},
        'H_AEHC': {'long_name': 'cumulative atmospheric boundary layer exceedance '
                                'heat content', 'units': 'PJ yr^-1',
                   'metric_type': 'compound'},
        'Nhours': {'long_name': 'daily exposure time', 'units': 'h'},
        'h_avg': {'long_name': 'average daily exposure time (DET)', 'units': 'h',
                  'metric_type': 'basic'},
        't_hfirst': {'long_name': 'daily hour of first exceedance', 'units': 'h',
                     'metric_type': 'basic'},
        't_hlast': {'long_name': 'daily hour of last exceedance', 'units': 'h',
                    'metric_type': 'basic'},
        't_hmax': {'long_name': 'daily hour of maximum exceedance', 'units': 'h',
                   'metric_type': 'basic'},
        'h_rise_avg': {'long_name': 'average daily exposure rise duration', 'units': 'h',
                       'metric_type': 'basic'},
        'h_set_avg': {'long_name': 'average daily exposure set duration', 'units': 'h',
                      'metric_type': 'basic'},
    }

    # add (A)GR indicators if necessary
    vname_dict = re.sub(r'(_GR|_AGR|_AF_CC|_AF|_CC|_ref|_p05|_p95)', '', vname)
    vattrs = attrs[vname_dict]
    if '_GR' in vname:
        vattrs['long_name'] = f'{vattrs["long_name"]} (GR)'
    elif '_AGR' in vname:
        vattrs['long_name'] = f'{vattrs["long_name"]} (AGR)'

    # add (CC) amplification if amplification factors are passed and set units to unity
    if 'AF' in vname and 'CC' not in vname:
        vattrs['long_name'] = f'{vattrs["long_name"]} amplification'
        vattrs['units'] = '1'
    elif 'AF_CC' in vname:
        vattrs['long_name'] = f'{vattrs["long_name"]} CC amplification'
        vattrs['units'] = '1'
    # ref and CC values
    elif '_ref' in vname:
        vattrs['long_name'] = f'{vattrs["long_name"]} (for ref period)'
    elif '_CC' in vname:
        vattrs['long_name'] = f'{vattrs["long_name"]} (for CC period)'
    
    # add decadal-mean for decadal variables
    if dec:
        vattrs['long_name'] = f'decadal-mean {vattrs["long_name"]}'

    # add upper/lower spread estimator for spreads
    if spread == 'upper' or spread == 'lower':
        vattrs['long_name'] = f'{vattrs["long_name"]} {spread} spread estimator'
    elif spread:
        vattrs['long_name'] = f'{vattrs["long_name"]} {spread}'

    return vattrs


def get_global_attrs(level=None, period=''):
    """
    get attributes for TEA variables
    Args:
        level: name of TEA indicator level (DBV, CTP, DEC, AF)
        period: climatic time period

    Returns:
        attributes: dict with attributes
    """

    titles = {
        'daily_results': f'Daily basis variables (DBV)',
        'ctp_results': f'Climatic time period variables (CTP)',
        'decadal_results': f'Decadal-mean indicator variables (DEC)',
        'amplification_factors': f'Decadal-mean amplification factors (AF)'}

    descriptions = {
        'daily_results': f'Daily basis TEA indicator variables',
        'ctp_results': f'TEA indicators for annual climatic time period: {period}',
        'decadal_results': f'TEA decadal-mean indicator variables for climatic time period: '
                           f'{period}',
        'amplification_factors': f'TEA decadal-mean amplification factors for climatic time period:'
                                 f' {period}'}

    # define general global attributes
    institution = 'Wegener Center for Climate and Global Change, University of Graz'
    cf_conv = 'CF-1.12'
    contact = 'WegenerNet Team <wegnet@wegenernet.org>'

    vattrs = {'Conventions': cf_conv, 'title': titles[level], 'institution': institution,
              'contact': contact, 'description': descriptions[level]}

    return vattrs
