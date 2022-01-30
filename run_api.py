from flask import Flask, render_template, request
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
import re
import pandas as pd

app = Flask(__name__)

vocab_size = 10000
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"

df = pd.read_csv('Tweets.csv')

#Suppression des doublons
df = df.drop_duplicates('tweet_id')

# Remove neutral rows
df = df.drop(df[df.airline_sentiment == 'neutral' ].index)

# Cleaning data set
def remove_useless_words_in_text(txt):
    mentions = re.findall("@([a-zA-Z0-9_]{1,50})", txt)
    for mention in mentions:
        txt = txt.replace(mention, '')
    txt = txt.replace('@', '')
    return txt

df['text'] = df['text'].transform(remove_useless_words_in_text)
    
sentences = [] # headlines
labels = [] # labels 
training_size = 1200
for idx,row in df.iterrows():
    if row['airline_sentiment_confidence'] > 0.7:
        sentences.append(row['text'])
        if row['airline_sentiment'] == 'positive':
            labels.append(1)
        elif row['airline_sentiment'] == 'negative':
            labels.append(0)

training_sentences = sentences[0:training_size]

tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>")
tokenizer.fit_on_texts(training_sentences)


class Analyse:
    def __init__(self, name, title, text, img_link, img_link_1, img_link_2, img_link_3, img_link_4, link_wikipedia, link_video):
        self.name = name
        self.title = title
        self.text = text
        self.img_link = img_link
        self.img_link_1 = img_link_1
        self.img_link_2 = img_link_2
        self.img_link_3 = img_link_3
        self.img_link_4 = img_link_4
        self.link_wikipedia = link_wikipedia
        self.link_video = link_video


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def post_search():

    model = keras.models.load_model('/Users/hurelmartin/Git-Repo/tweet-sentiment-analyse/price_prediction_model.h5')




    phrase = request.form.get('phrase')


    new_sequences = tokenizer.texts_to_sequences(new_sentence)
    # padding the new sequences to make them have same dimensions
    new_padded = pad_sequences(new_sequences, maxlen = max_length,

                            padding = padding_type,
                            truncating = trunc_type)

    new_padded = np.array(new_padded)

    print(model.predict(new_padded))


    prediction = model.predict(phrase)

    print(prediction)
    

    # if(predict == 0):
    #     values = Star("Brown Dwarf", "Naine brune en français", "Une naine brune est, d'après la définition provisoire adoptée, en 2003, par l'Union astronomique internationale, un objet substellaire dont la vraie masse est inférieure à la masse minimale nécessaire à la fusion thermonucléaire de l'hydrogène mais supérieure à celle nécessaire à la fusion thermonucléaire du deutérium1, correspondant à une masse située entre 13 MJ et 75 MJ2. En d'autres termes, il s'agit d'un objet insuffisamment massif pour être considéré comme une étoile mais plus massif qu'une planète géante. Il y a un accord sur la limite supérieure en deçà de laquelle une naine brune ne peut entretenir la réaction de fusion nucléaire de l'hydrogène : moins de 0,07 masse solaire pour une composition chimique solaire. La limite inférieure quant à elle ne fait pas unanimité ; un critère couramment retenu est la capacité à fusionner le deutérium, soit environ 13 masses MJ.La classification spectrale des naines brunes a motivé une extension de celle des étoiles : elles ont pour type spectral M, L, T voire Y pour les plus froides.L'énergie lumineuse d'une naine brune est quasi exclusivement tirée de l'énergie potentielle gravitationnelle, transformée en énergie interne par contraction, contrairement à une étoile de la séquence principale qui tire son énergie des réactions nucléaires. La contraction s'achève lorsque se produit la dégénérescence de la matière, la naine brune a alors un diamètre de l'ordre de celui de la planète Jupiter. En l'absence d'autre source d'énergie, une naine brune se refroidit au cours de son existence, et parcourt les types spectraux M, L et T ; ceci diffère d'une étoile de la séquence principale dont la température effective et le type spectral restent sensiblement constants.Bien que leur existence fût postulée dès les années 1960, c'est seulement depuis le milieu des années 1990 qu'on a pu établir leur existence.",
    #                   "https://upload.wikimedia.org/wikipedia/commons/e/e0/Artist%E2%80%99s_conception_of_a_brown_dwarf_like_2MASSJ22282889-431026.jpg", "https://images.theconversation.com/files/106259/original/image-20151216-25600-tbnhyq.jpg?ixlib=rb-1.1.0&q=45&auto=format&w=1356&h=668&fit=crop", "https://media1.s-nbcnews.com/j/MSNBC/Components/Photo/_new/120123-space-sun-10a.fit-760w.jpg", "https://cdn.pixabay.com/photo/2017/07/25/19/27/star-2539245_960_720.jpg", "https://cdn.pixabay.com/photo/2017/08/05/18/24/star-2584986_960_720.jpg", "https://fr.wikipedia.org/wiki/Naine_brune", "4zKVx29_A1w")
    # elif(predict == 1):
    #     values = Star("Red Dwarf", "Naine rouge en français", "En astronomie, une étoile rouge de la séquence principale, appelée communément naine rouge, est une étoile de type spectral M V (lire « M cinq »), c'est-à-dire une étoile appartenant à la séquence principale (classe de luminosité V) de type spectral M (étoile rouge). Les étoiles K dites tardives (naines orange les plus froides) sont parfois incluses parmi les naines rouges. Ces étoiles sont peu massives et de température peu élevée. Ayant une masse comprise entre 0,075 et 0,4 masse solaire (M☉) et une température inférieure à 4 000 K en surface, ce sont des étoiles peu lumineuses, les plus grosses d'entre elles émettant de l'ordre de 10 % de la luminosité solaire. En dessous de 0,08 M☉, on a affaire à un objet substellaire, à une naine brune ou à une planète géante gazeuse. La limite entre étoile naine rouge et naine brune de type spectral M est généralement au niveau du type M 6.5.Les naines rouges seraient de loin les étoiles les plus nombreuses de l'Univers1. Les modèles stellaires actuels les décrivent comme entièrement convectives, c'est-à-dire que l'hydrogène est constamment brassé par convection dans l'ensemble de l'étoile de sorte que l'hélium issu de la réaction proton-proton au cœur de l'astre ne peut s'y accumuler. Les naines rouges pourraient ainsi briller de façon relativement constante pendant des centaines de milliards d'années2, c'est-à-dire plusieurs dizaines de fois l'âge de l'Univers, ce qui signifie que toutes les naines rouges actuelles n'en seraient qu'au début de leur existence.",
    #                   "https://assets.newatlas.com/dims4/default/38bbef3/2147483647/strip/true/crop/1200x800+0+80/resize/1200x800!/quality/90/?url=http%3A%2F%2Fnewatlas-brightspot.s3.amazonaws.com%2Farchive%2Fred-dwarf-water-world-1.jpg", "https://images.theconversation.com/files/106259/original/image-20151216-25600-tbnhyq.jpg?ixlib=rb-1.1.0&q=45&auto=format&w=1356&h=668&fit=crop", "https://media1.s-nbcnews.com/j/MSNBC/Components/Photo/_new/120123-space-sun-10a.fit-760w.jpg", "https://cdn.pixabay.com/photo/2017/07/25/19/27/star-2539245_960_720.jpg", "https://cdn.pixabay.com/photo/2017/08/05/18/24/star-2584986_960_720.jpg", "https://fr.wikipedia.org/wiki/Red_Dwarf", "LS-VPyLaJFM")
    # elif(predict == 2):
    #     values = Star("White Dwarf", "Naine blanche en français",
    #                   "Une naine blanche1, 2, 3 est un objet céleste de forte densité, issu de l'évolution d'une étoile de masse modérée(de trois à quatre masses solaires au maximum4) après la phase où se produisent des réactions thermonucléaires. Cet objet a alors une taille très petite, comparativement à une étoile, et conserve longtemps une température de surface élevée, d'où son nom de « naine blanche ».Une naine blanche possède typiquement une masse inférieure quoique comparable à celle du Soleil pour un volume similaire à celui de la Terre. Sa masse volumique est ainsi de l’ordre d’une tonne par centimètre cube, plusieurs dizaines de milliers de fois plus élevée que celle des matériaux observés sur Terre. Sa température de surface, qui peut dépasser au départ 100 000 kelvins, provient de la chaleur emmagasinée par son étoile parente, chaleur dont le transfert thermique est très lent du fait de la faible surface de l'astre5. C'est aussi du fait de cette faible surface que, malgré sa température élevée, la luminosité d'une naine blanche reste limitée à une valeur de l’ordre d’un millième de luminosité solaire, et décroît au cours du temps.Début 2009, le projet Research Consortium on Nearby Stars dénombre huit naines blanches dans les cent systèmes stellaires les plus proches du Système solaire6, mais étant donné la rareté des étoiles de grande masse, elles représentent le destin de 96 % des étoiles de notre galaxie7.Du fait de l'évolution de leur étoile parente (dictée par sa masse), les naines blanches existant aujourd'hui sont habituellement composées de carbone et d'oxygène. Quand l'étoile parente est suffisamment massive(probablement entre huit et dix masses solaires), il est possible qu'elle donne naissance à une naine blanche sans carbone, mais comprenant du néon et du magnésium en plus de l'oxygène8. Il est également possible qu'une naine blanche soit principalement composée d'hélium9, 10, si son étoile parente a été sujette à un transfert de matière dans un système binaire. Dans ces deux cas, la naine blanche correspond au cœur mis à nu de l'étoile parente, alors que les couches externes de celle-ci ont été expulsées et ont formé une nébuleuse planétaire. Il n'existe pas de naines blanches issues d'étoiles de moins d'une demi-masse solaire, car la durée de vie de celles-ci est supérieure à l'âge de l'Univers. Ces étoiles-là évolueront selon toute vraisemblance en des naines blanches composées d'hélium11.La structure interne d'une naine blanche est déterminée par l'équilibre entre la gravité et les forces de pression, ici produite par un phénomène de mécanique quantique appelé pression de dégénérescence. Les calculs indiquent que cet équilibre ne peut subsister pour des astres de plus de 1, 4 masse solaire({\displaystyle M_{\odot}}M_{\odot}). Il s'agit donc de la masse maximale que peut posséder une naine blanche lors de sa formation ou de son évolution. C'est cette masse maximale qui fixe la masse maximale initiale de huit masses solaires que peut avoir une étoile pour que celle-ci évolue en naine blanche, la différence entre ces deux valeurs correspondant aux pertes de masse subies par l'étoile lors de son évolution. Une naine blanche isolée est un objet d'une très grande stabilité, qui va simplement se refroidir au cours du temps pour, à très long terme, devenir une naine noire. Si par contre une naine blanche possède un compagnon stellaire, elle pourra éventuellement interagir avec ce compagnon, formant ainsi une variable cataclysmique. Elle se manifestera sous différentes formes suivant le processus d'interaction : nova classique, source super molle, nova naine, polaire ou polaire intermédiaire. Ces interactions tendent à faire augmenter la masse de la naine blanche par accrétion. Dans l'éventualité où celle-ci atteint la masse critique de 1, 4 {\displaystyle M_{\odot}}M_{\odot}(par accrétion voire par collision avec une autre naine blanche), elle achèvera sa vie de façon paroxystique en une gigantesque explosion thermonucléaire appelée supernova de type Ia5, 12.En spectroscopie, les naines blanches forment la classe D de la classification spectrale des étoiles et de leurs résidus. Elles sont réparties entre plusieurs sous-classes — DA13, DB14, DC15, DO16, DQ17 et DZ18 — en fonction des caractéristiques de leur spectre.", "https://aasnova.org/wp-content/uploads/2017/05/fig1-12.jpg", "https://images.theconversation.com/files/106259/original/image-20151216-25600-tbnhyq.jpg?ixlib=rb-1.1.0&q=45&auto=format&w=1356&h=668&fit=crop", "https://media1.s-nbcnews.com/j/MSNBC/Components/Photo/_new/120123-space-sun-10a.fit-760w.jpg", "https://cdn.pixabay.com/photo/2017/07/25/19/27/star-2539245_960_720.jpg", "https://cdn.pixabay.com/photo/2017/08/05/18/24/star-2584986_960_720.jpg", "https://fr.wikipedia.org/wiki/Naine_blanche", "qsN1LglrX9s")
    # elif(predict == 3):
    #     values = Star("Main Sequence", "Séquence principale en français",
    #                   "En astronomie, la séquence principale est une bande continue et bien distincte d'étoiles qui apparaissent sur des diagrammes où l'abscisse est l'indice de couleur B-V note 1 et l'ordonnée la luminosité ou, en sens inverse, la magnitude absolue des étoiles. Ces diagrammes couleur-luminosité sont connus sous le nom de « diagrammes de Hertzsprung-Russell », d'après leur co-inventeurs Ejnar Hertzsprung et Henry Norris Russell. Les étoiles figurant dans cette bande sont connues sous le nom d’étoiles de la série principale, ou « étoiles naines »1,2. Ainsi, environ 90 % des étoiles observées au-dessus de 0,5 M☉ sont sur la séquence principale[réf. nécessaire].La séquence principale désigne aussi le stade principal de l'évolution d'une étoile : c'est pendant cette période que ses caractéristiques correspondent à celles de la séquence principale du diagramme Hertzsprung-Russell et qu'elle s'y trouve effectivement représentée.La proportion élevée d'étoiles sur la séquence principale est due au fait que cette séquence correspond à la phase de fusion de l'hydrogène en hélium, laquelle dure la majeure partie de la durée de vie totale de l'étoile (en raison de la prépondérance de l'hydrogène dans la composition initiale, et aussi parce que la fusion d'hydrogène en hélium est la plus exoénergétique des réactions de fusion nucléaire).", "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/Sirius_A_and_B_artwork.jpg/1200px-Sirius_A_and_B_artwork.jpg", "https://images.theconversation.com/files/106259/original/image-20151216-25600-tbnhyq.jpg?ixlib=rb-1.1.0&q=45&auto=format&w=1356&h=668&fit=crop", "https://media1.s-nbcnews.com/j/MSNBC/Components/Photo/_new/120123-space-sun-10a.fit-760w.jpg", "https://cdn.pixabay.com/photo/2017/07/25/19/27/star-2539245_960_720.jpg", "https://cdn.pixabay.com/photo/2017/08/05/18/24/star-2584986_960_720.jpg", "https://fr.wikipedia.org/wiki/S%C3%A9quence_principale", "N7q1KJsWtx8")
    # elif(predict == 4):
    #     values = Star("Supergiant", "Supergiant en français", "Dans le diagramme de Hertzsprung-Russell, les supergéantes occupent le haut du diagramme. Dans la classification MKK, les supergéantes peuvent être de classe Ia (supergéantes très lumineuses) ou Ib (supergéantes moins lumineuses). Typiquement, la magnitude bolométrique absolue d'une supergéante est comprise entre -5 et -12.La masse des supergéantes varie entre 10 et 70 masses solaires et leur luminosité de 30 000 à plusieurs centaines de milliers de fois la luminosité solaire. Elles varient fortement en taille, entre 30 et 500, voire plus de 1 000 rayons solaires. La loi de Stefan-Boltzmann implique que la surface plus froide des supergéantes rouges rayonne moins d'énergie par unité de surface que les supergéantes bleues ; pour une luminosité donnée, les supergéantes rouges sont donc plus grandes que les supergéantes bleues.Les supergéantes existent dans tous les types spectraux, depuis les jeunes supergéantes bleues de classe O aux supergéantes rouges de classe M, fortement évoluées.La modélisation des supergéantes est un domaine de recherche toujours d'actualité ; elle est compliquée par divers facteurs comme la prise en compte de la perte de masse par l'étoile au cours du temps. Plutôt que de modéliser des étoiles individuelles, la tendance actuelle est à la modélisation d'amas d'étoiles et à la comparaison des modèles résultants avec la distribution observée de supergéantes dans les galaxies comme dans les nuages de Magellan.",
    #                   "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7f/VX_Sagittarii_Red_Supergiant_Star.png/1200px-VX_Sagittarii_Red_Supergiant_Star.png", "https://images.theconversation.com/files/106259/original/image-20151216-25600-tbnhyq.jpg?ixlib=rb-1.1.0&q=45&auto=format&w=1356&h=668&fit=crop", "https://media1.s-nbcnews.com/j/MSNBC/Components/Photo/_new/120123-space-sun-10a.fit-760w.jpg", "https://cdn.pixabay.com/photo/2017/07/25/19/27/star-2539245_960_720.jpg", "https://cdn.pixabay.com/photo/2017/08/05/18/24/star-2584986_960_720.jpg", "https://en.wikipedia.org/wiki/Supergiant_star", "lTwzM6hM5ww")
    # elif(predict == 5):
    #     values = Star("Hypergiant", "Hypergiant en français", "Une hypergéante jaune est une étoile massive à l'atmosphère étendue et de classe spectrale variant de la fin de la classe A jusqu'au début de la classe K sur le diagramme Hertzsprung-Russell (HR). Sa masse initiale équivaut à 20 à 50 masses solaires, mais à ce stade, elle a pu perdre jusqu'à la moitié de cette masse2. Jusqu'ici, seule une poignée d'entre elles sont répertoriées dans notre galaxie.Parfois appelées hypergéantes froides en comparaison avec les étoiles faisant partie des classes O et B, et parfois hypergéantes tièdes en comparaison avec les supergéantes rouges3,4, les hypergéantes jaunes comptent parmi les étoiles les plus lumineuses jamais observées, avec une magnitude absolue (MV) se situant aux environs de -9. Elles sont aussi parmi les étoiles les plus volumineuses5 et les plus rares.",
    #                   "http://cdn.eso.org/images/screen/eso1409b.jpg", "https://images.theconversation.com/files/106259/original/image-20151216-25600-tbnhyq.jpg?ixlib=rb-1.1.0&q=45&auto=format&w=1356&h=668&fit=crop", "https://media1.s-nbcnews.com/j/MSNBC/Components/Photo/_new/120123-space-sun-10a.fit-760w.jpg", "https://cdn.pixabay.com/photo/2017/07/25/19/27/star-2539245_960_720.jpg", "https://cdn.pixabay.com/photo/2017/08/05/18/24/star-2584986_960_720.jpg", "https://en.wikipedia.org/wiki/Hypergiant", "bWYuch-s61A")

    # return render_template('stars.html', prediction=values)


if __name__ == '__main__':
    app.run(debug=True)
