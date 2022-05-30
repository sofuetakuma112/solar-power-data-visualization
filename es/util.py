import datetime


def isoformat2dt(jptime):
    return datetime.datetime.fromisoformat(jptime)


def sortDocsByKey(docs, key):
    return sorted(  # datetimeでソート
        docs,
        key=lambda doc: isoformat2dt(extractFieldsFromDoc(doc, key)) + datetime.timedelta(hours=9),
    )


def isoformats2dt(isoformats):
    return list(
        map(
            lambda isoformat: isoformat2dt(isoformat),
            isoformats,
        )
    )


def extractFieldsFromDoc(doc, key):
    return doc["_source"][key]


def extractFieldsFromDocs(docs, key):
    return list(map(lambda doc: extractFieldsFromDoc(doc, key), docs))
