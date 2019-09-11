<!--####################################################################################
# This file creates the markdown documentation for the woonfraude detection repo.      #
#                                                                                      #
# Created by Thomas Jongstra 2019 - for the Municipality of Amsterdam                  #
#####################################################################################-->

# Woonfraude Detection
Project by: Swaan Dekkers, Saban Karki, Ammar Khawaja, Geert Reeken, Felicien Veldema, Thomas Jongstra

#### Hoofdvraag

Is woonfraude te voorspellen met data?

#### Subvragen

* Is op basis van een melding de kans op woonfraude te voorspellen?
* Is te achterhalen welke zaken uit meldingen voortkomen, en welke niet?
* Is dit te doen met een (aan handhavers) uitlegbaar algoritme?

## Package Structuur

De code in dit pakket (package) is onderverdeeld in 3 subpaketten (subpackages):

* **codebase:** bevat alle kern-functionaliteiten om dataset specifieke operaties uit te voeren (datasets.py) data op te schonen (clean.py), features te extracten (extract_features.py) en om machine learning modellen te bouwen (build_model.py). [TODO: dashboard_link.py in **dashboard** subpakket zetten en beschrijven?]

* **dashboard:** bevat de code om een interactief dashboard te genereren (dashboard.py). Hiernaast bevat dit subpakket ook mock-up data (mockup_dataset.csv, mockup_dataset_proactief.csv,mockup_dataset_unsupervised.csv) om het dashboard lokaal te kunnen testen, zonder dat hier gevoelige data voor nodig is. Ten slotte bevat de directory "assets" alle benodigde opmaak-bestanden en iconen om het dashboard er mooi uit te laten zien.

* **notebooks:** hierin staan Jupyter notebooks, waarmee de code uit de **codebase** uitgevoerd kan worden. Ze maken het makkelijk om de functies uit de codebase op high-level aan te sturen. De basis notebook (base.ipynb) importeert alle functies uit de codebase zodat hier direct me gewerkt kan worden. Deze notebook kan als basis gebruikt worden voor nieuwe analyses of modeluitwerkingen. De master notebook (master.ipynb) implementeert een volledig stappenplan (datasets inladen, data opschonen, features extracten, model trainen) om een voorspelmodel te maken dat gebruikt kan worden voor voorspellingen op nieuwe adressen (bijvoorbeeld binnen het dashboard).