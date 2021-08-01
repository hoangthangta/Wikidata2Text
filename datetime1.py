from datetime import *


def convert_datetime_ISO8601(datetime_label):
    try:
        dt = datetime.strptime(datetime_label, '%Y-%m-%dT%H:%M:%SZ')
        return dt
    except Exception as e:
        
        try:
            temp = datetime_label.split('T')
            temp1 = temp[0].split('-')
            if (temp1[1] == '00'):
                temp1[1] = '01'
            if (temp1[2] == '00'):
                temp1[2] = '01'
            label = '-'.join(e for e in temp1)
            label = label + 'T' + temp[1]  
            dt = datetime.strptime(label, '%Y-%m-%dT%H:%M:%SZ')
            return dt       
        except Exception as e:
            pass
    return False

   

#date_time_str = '1910-00-00T00:00:00Z'
#date_time_obj = convert_datetime_ISO8601(date_time_str)
#date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%dT%H:%M:%SZ')
#print('Date:', date_time_obj.date())
#print('Time:', date_time_obj.time())
#print('Date-time:', date_time_obj)

