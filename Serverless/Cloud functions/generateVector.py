rom google.cloud import storage
import pandas as pd
import io
import csv

def generateVector(event, context):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(event['bucket'])
    filename = event['name']

    blobfile = bucket.blob(filename)
    blobfile = blobfile.download_as_string()
    blobfile = blobfile.decode('utf-8')

    try:
        doc = []
        for i in blobfile:
            words = str(i).split(" ")
            for w in words:
                w = w.replace("b'", "")
                w = w.replace("\\n'", "")
                w = w.replace('b"', "")
                if w:
                    doc.append(w.lower())
    
        # load stop words in a list
        stop_list = ["able", "about", "above", "abroad", "according", "accordingly", "across",
                    "actually", "adj", "after", "afterwards", "again", "against", "ago",
                    "ahead", "aint", "all", "allow", "allows", "almost", "alone", "along",
                    "alongside", "already", "also", "although", "always", "am", "amid", "amidst", "among",
                    "amongst", "an", "and", "another", "any", "anybody", "anyhow", "anyone", "anything",
                    "anyway", "anyways", "anywhere", "apart", "appear", "appreciate", "appropriate", "are", "arent",
                    "around", "as", "as", "aside", "ask", "asking", "associated", "at", "available", "away", "awfully",
                    "back", "backward", "backwards", "be", "became", "because", "become", "becomes", "becoming",
                    "been", "before", "beforehand", "begin", "behind", "being", "believe", "below", "beside", "besides",
                    "best", "better", "between", "beyond", "both", "brief", "but", "by", "came", "can", "cannot", "cant",
                    "cant", "caption", "cause", "causes", "certain", "certainly", "changes", "clearly", "cmon", "co", "co",
                    "com", "come", "comes", "concerning", "consequently", "consider", "considering", "contain",
                    "containing", "contains", "corresponding", "could", "couldnt", "course", "cs", "currently", "dare",
                    "darent", "definitely", "described", "despite", "did", "didnt", "different", "directly", "do", "does",
                    "doesnt", "doing", "done", "dont", "down", "downwards", "during", "each", "edu", "eg", "eight",
                    "eighty", "either", "else", "elsewhere", "end", "ending", "enough", "entirely", "especially", "et",
                    "etc", "even", "ever", "evermore", "every", "everybody", "everyone", "everything", "everywhere", "ex",
                    "exactly", "example", "except", "fairly", "far", "farther", "few", "fewer", "fifth", "first", "five",
                    "followed", "following", "follows", "for", "forever", "former", "formerly", "forth", "forward",
                    "found", "four", "from", "further", "furthermore", "get", "gets", "getting", "given", "gives", "go",
                    "goes", "going", "gone", "got", "gotten", "greetings", "had", "hadnt", "half", "happens", "hardly",
                    "has", "hasnt", "have", "havent", "having", "he", "hed", "hell", "hello", "help", "hence", "her",
                    "here", "hereafter", "hereby", "herein", "heres", "hereupon", "hers", "herself", "hes", "hi", "him",
                    "himself", "his", "hither", "hopefully", "how", "howbeit", "however", "hundred", "id", "ie", "if",
                    "ignored", "ill", "im", "immediate", "in", "inasmuch", "inc", "inc", "indeed", "indicate", "indicated",
                    "indicates", "inner", "inside", "insofar", "instead", "into", "inward", "is", "isnt", "it", "itd",
                    "itll", "its", "its", "itself", "ive", "just", "k", "keep", "keeps", "kept", "know", "known", "knows",
                    "last", "lately", "later", "latter", "latterly", "least", "less", "lest", "let", "lets", "like",
                    "liked", "likely", "likewise", "little", "look", "looking", "looks", "low", "lower", "ltd", "made",
                    "mainly", "make", "makes", "many", "may", "maybe", "maynt", "me", "mean", "meantime", "meanwhile",
                    "merely", "might", "mightnt", "mine", "minus", "miss", "more", "moreover", "most", "mostly", "mr",
                    "mrs", "much", "must", "mustnt", "my", "myself", "name", "namely", "nd", "near", "nearly", "necessary",
                    "need", "neednt", "needs", "neither", "never", "neverf", "neverless", "nevertheless", "new", "next",
                    "nine", "ninety", "no", "nobody", "non", "none", "nonetheless", "noone", "no-one", "nor", "normally",
                    "not", "nothing", "notwithstanding", "novel", "now", "nowhere", "obviously", "of", "off", "often",
                    "oh", "ok", "okay", "old", "on", "once", "one", "ones", "ones", "only", "onto", "opposite", "or",
                    "other", "others", "otherwise", "ought", "oughtnt", "our", "ours", "ourselves", "out", "outside",
                    "over", "overall", "own", "particular", "particularly", "past", "per", "perhaps", "placed", "please",
                    "plus", "possible", "presumably", "probably", "provided", "provides", "que", "quite", "qv", "rather",
                    "rd", "re", "really", "reasonably", "recent", "recently", "regarding", "regardless", "regards",
                    "relatively", "respectively", "right", "round", "said", "same", "saw", "say", "saying", "says",
                    "second", "secondly", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves",
                    "sensible", "sent", "serious", "seriously", "seven", "several", "shall", "shant", "she", "shed",
                    "shell", "shes", "should", "shouldnt", "since", "six", "so", "some", "somebody", "someday", "somehow",
                    "someone", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specified",
                    "specify", "specifying", "still", "su", "such", "sup", "sure", "take", "taken", "taking", "tell",
                    "tends", "th", "than", "thank", "thanks", "thanx", "that", "thatll", "thats", "thats", "thatve", "the",
                    "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered",
                    "therefore", "therein", "therell", "therere", "theres", "theres", "thereupon", "thereve", "these",
                    "they", "theyd", "theyll", "theyre", "theyve", "thing", "things", "think", "third", "thirty", "this",
                    "thorough", "thoroughly", "those", "though", "three", "through", "throughout", "thru", "thus", "till",
                    "to", "together", "too", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying", "ts",
                    "twice", "two", "un", "under", "underneath", "undoing", "unfortunately", "unless", "unlike",
                    "unlikely", "until", "unto", "up", "upon", "upwards", "us", "use", "used", "useful", "uses", "using",
                    "usually", "v", "value", "various", "versus", "very", "via", "viz", "vs", "want", "wants", "was",
                    "wasnt", "way", "we", "wed", "welcome", "well", "well", "went", "were", "were", "werent", "weve",
                    "what", "whatever", "whatll", "whats", "whatve", "when", "whence", "whenever", "where", "whereafter",
                    "whereas", "whereby", "wherein", "wheres", "whereupon", "wherever", "whether", "which", "whichever",
                    "while", "whilst", "whither", "who", "whod", "whoever", "whole", "wholl", "whom", "whomever", "whos",
                    "whose", "why", "will", "willing", "wish", "with", "within", "without", "wonder", "wont", "would",
                    "wouldnt", "yes", "yet", "you", "youd", "youll", "your", "youre", "yours", "yourself", "yourselves",
                    "youve", "zero", "a", "hows", "i", "whens", "whys", "www", "amount", "bill", "bottom", "call",
                    "computer", "con", "couldnt", "cry", "de", "describe", "detail", "due", "eleven", "empty", "fifteen",
                    "fifty", "fill", "find", "fire", "forty", "front", "full", "give", "hasnt", "herse", "himse",
                    "interest", "itself", "mill", "move", "part", "put", "show", "side", "sincere", "sixty", "system",
                    "ten", "thick", "thin", "top", "twelve", "twenty", "abst", "accordance", "act", "added", "adopted",
                    "affected", "affecting", "affects", "ah", "announce", "anymore", "apparently", "approximately", "aren",
                    "arent", "arise", "auth", "beginning", "beginnings", "begins", "biol", "briefly", "ca", "date", "ed",
                    "effect", "et-al", "ff", "fix", "gave", "giving", "heres", "hes", "hid", "home", "id", "im",
                    "immediately", "importance", "important", "index", "information", "invention", "itd", "keys", "kg",
                    "km", "largely", "lets", "line", "ll", "means", "mg", "million", "ml", "mug", "na", "nay",
                    "necessarily", "nos", "noted", "obtain", "obtained", "omitted", "ord", "owing", "page", "pages",
                    "poorly", "possibly", "potentially", "pp", "predominantly", "present", "previously", "primarily",
                    "promptly", "proud", "quickly", "ran", "readily", "ref", "refs", "related", "research", "resulted",
                    "resulting", "results", "run", "sec", "section", "shed", "shes", "showed", "shown", "showns", "shows",
                    "significant", "significantly", "similar", "similarly", "slightly", "somethan", "specifically",
                    "state", "states", "stop", "strongly", "substantially", "successfully", "sufficiently", "suggest",
                    "thered", "thereof", "therere", "thereto", "theyd", "theyre", "thou", "thoughh", "thousand", "through",
                    "til", "tip", "ts", "ups", "usefully", "usefulness", "ve", "vol", "vols", "wed", "whats", "wheres",
                    "whim", "whod", "whos", "widely", "words", "world", "youd", "youre"]
    
        # calculate levenshtein distance
        curr = ""
        dist = 0
        header = ['Current_Word ', 'Next_Word ', 'Levenshtein_distance']
        output = []
        for y in range(len(doc)):
            if curr == "":
                curr = doc[y]
            else:
                dist = dist + 1
                if doc[y].lower() not in stop_list:
                    rows = [curr+ " ", doc[y]+ " ", str(dist)]
                    output.append(rows)
                    dist = 0
                    curr = doc[y]

        df_new = pd.DataFrame(output, columns=header)
        
        bucket1 = storage_client.get_bucket("traindatab00882286")
        df = pd.read_csv(io.BytesIO(
                 bucket1.blob(blob_name = 'trainVector.csv').download_as_string()) ,
                 encoding='UTF-8',
                 sep=' ')

        
        frames = [df, df_new]
        result = pd.concat(frames)

        result.to_csv('/tmp/trainVector.csv')

        blob = bucket1.blob('trainVector.csv')
        blob.upload_from_filename("/tmp/trainVector.csv")

                   
    except Exception as e:
        raise e
   