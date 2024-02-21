import datetime


def get_total_seconds(dt):
    epoch_time = datetime.datetime(1970, 1, 1)
    return (dt - epoch_time).total_seconds()


def mask_from_into_dt(mask_from, y, m, d):
    mask_from_hour, mask_from_minute = mask_from.split(":")
    return datetime.datetime(
        int(y),
        int(m),
        int(d),
        int(mask_from_hour),
        int(mask_from_minute),
        int(0),
    )


def mask_to_into_dt(mask_to, y, m, d):
    mask_to_hour, mask_to_minute = mask_to.split(":")
    if int(mask_to_hour) == 24:
        return datetime.datetime(
            int(y),
            int(m),
            int(d),
            int(0),
            int(mask_to_minute),
            int(0),
        ) + datetime.timedelta(days=1)
    else:
        return datetime.datetime(
            int(y),
            int(m),
            int(d),
            int(mask_to_hour),
            int(mask_to_minute),
            int(0),
        )
