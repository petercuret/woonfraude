<!--####################################################################################
# This file creates the markdown documentation for the woonfraude detection repo.      #
#                                                                                      #
# Created by Thomas Jongstra 2019 - for the Municipality of Amsterdam                  #
#####################################################################################-->

# Woonfraude Detectie
Project by: Swaan Dekkers, Saban Karki, Ammar Khawaja, Geert Reeken, Felicien Veldema, Thomas Jongstra


#### Hoofdvraag

Is woonfraude te voorspellen met data?

#### Subvragen

* Is op basis van een melding de kans op woonfraude te voorspellen?
* Is te achterhalen welke zaken uit meldingen voortkomen, en welke niet?
* Is dit te doen met een (aan handhavers) uitlegbaar algoritme?


## Pakket Structuur

De code in dit pakket (package) is onderverdeeld in 3 subpaketten (subpackages):

* **codebase:** bevat alle kern-functionaliteiten om dataset specifieke operaties uit te voeren (datasets.py) data op te schonen (clean.py), features te extracten (extract_features.py) en om machine learning modellen te bouwen (build_model.py). [TODO: dashboard_link.py in **dashboard** subpakket zetten en beschrijven?]

* **dashboard:** bevat de code om een interactief dashboard te genereren (dashboard.py). Hiernaast bevat dit subpakket ook mock-up data (mockup_dataset.csv, mockup_dataset_proactief.csv,mockup_dataset_unsupervised.csv) om het dashboard lokaal te kunnen testen, zonder dat hier gevoelige data voor nodig is. Ten slotte bevat de directory "assets" alle benodigde opmaak-bestanden en iconen om het dashboard er mooi uit te laten zien.

* **notebooks:** hierin staan Jupyter notebooks, waarmee de code uit de **codebase** uitgevoerd kan worden. Ze maken het makkelijk om de functies uit de codebase op high-level aan te sturen. De basis notebook (base.ipynb) importeert alle functies uit de codebase zodat hier direct me gewerkt kan worden. Deze notebook kan als basis gebruikt worden voor nieuwe analyses of modeluitwerkingen. De master notebook (master.ipynb) implementeert een volledig stappenplan (datasets inladen, data opschonen, features extracten, model trainen) om een voorspelmodel te maken dat gebruikt kan worden voor voorspellingen op nieuwe adressen (bijvoorbeeld binnen het dashboard).

Direct in de root van de repository zitten ook enkele bestanden:

* **.gitignore:** dit bestand geeft aan welke bestanden niet relevant zijn voor deze git repository (bijv. bestanden met passwords, of persoonlijke notebooks die voor eigen onderzoek worden gebruikt). Dankzij de .gitignore file worden deze bestanden bij het synchorniseren met git genegeerd.
* **LICENSE:** een bestand waarin de licentievorm (MIT license), waaronder deze code openbaar wordt gemaakt, staat omschreven.
* **README.MD:** dit bestand genereert de readme-tekst die op de Github voorpagina van deze repository wordt vertoond. Wellicht lees je die tekst nu!
* **requirements.txt:** een lijst met Python paketten waar het woonfraude pakket afhankelijk van is.
* **whitelist.txt:** een lijst met websites die voor dit pakket toegankelijk moeten zijn; o.a. om de kaart op het dashboard weer te kunnen geven, maar ook om actief aan deze repository bij te kunnen dragen.


## Lokale Installatie

[TODO: WELLICHT PAKKET OP PYPI ZETTEN EN PIP INSTALLATIE MOGELIJK MAKEN?]

#### Download de code uit deze repository:
    git clone https://github.com/jongstra/woonfraude.git

#### Ga met de terminal de nieuwe gedownloade directory in:
    cd woonfraude

#### Maak een lokale virtuele Python omgeving aan:
    virtualenv --python=$(which python3) venv

#### Activeer deze virtuele omgeving (doe dit iedere keer dat je met de code werkt):
    source venv/bin/activate

#### Installeer de benodigde pakketten in deze virtuele omgeving:
    pip install requirements.txt

Nu zou je klaar moeten zijn om met dit pakket te werken.


## Pakkket Werking

#### Om zelf een model te trainen om een analyse te doen, kun je een persoonlijke folder aanmaken voor je eigen notebooks:
    mkdir notebooks_prive

#### Je kunt nu de base notebook (in notebooks/base.ipynb) zelf in deze folder kopieren:
    cp notebooks/base.ipynb notebooks_prive/mijn_notebook.ipynb

#### Je kunt nu zelf deze noteobok aanpassen voor je eigen analyse/onderzoeksvraag. Start hiervoor Jupyter Lab op:
    jupyter lab

Navigeer nu naar je eigen notebook (mijn_notebook.ipynb) en ga aan de slag!

#### Om de volledige voorgebouwde machine learning straat door te lopen kun je ook de master notebook (notebooks/master.ipynb) kopieren naar je eigen notebooks_prive folder:
    cp notebooks/master.ipynb notebooks_prive/master.ipynb

Als je deze notebook opent in Jupyter Lab en alle cellen uitvoert, dan heb je uiteindelijk een voorgetraind voorspelmodel staan in de folder waar je notebook ook staat.