import cdsapi

if __name__ == "__main__":
    # YEARS = ["%d" % (y) for y in range(1956, 2023)]
    YEARS = ['2025']
    VARS = ['10m_u_component_of_wind',
            '10m_v_component_of_wind',
            '2m_dewpoint_temperature',
            '2m_temperature',
            'surface_pressure',
            'total_precipitation']
    MONTHS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12',]

    for yr in YEARS:
        for var in VARS:
            for mon in MONTHS:
                request = {
                    'variable': [var],
                    'year': f'{yr}',
                    'month': [mon],
                    'day': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                        '13', '14', '15',
                        '16', '17', '18',
                        '19', '20', '21',
                        '22', '23', '24',
                        '25', '26', '27',
                        '28', '29', '30',
                        '31',
                    ],
                    'time': [
                        '00:00', '01:00', '02:00',
                        '03:00', '04:00', '05:00',
                        '06:00', '07:00', '08:00',
                        '09:00', '10:00', '11:00',
                        '12:00', '13:00', '14:00',
                        '15:00', '16:00', '17:00',
                        '18:00', '19:00', '20:00',
                        '21:00', '22:00', '23:00',
                    ],
                    'area': [
                        49.5, 9, 46,
                        17.5,
                    ],
                    'format': 'netcdf',
                }

                client = cdsapi.Client()
                client.retrieve("reanalysis-era5-land", request,
                                f'ERA5Land_{yr}-{mon}_{var}.nc')
