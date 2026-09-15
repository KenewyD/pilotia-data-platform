# Pitch de présentation — PILOT'IA

## Pitch 30 secondes

À la suite de mon échange sur les problématiques de pilotage, j'ai développé un prototype
qui transforme des données d'activité synthétiques en indicateurs opérationnels.
L'application couvre la qualité des données, les délais de service, l'adéquation charge-ressources,
la prévision d'activité et la détection d'anomalies.

L'objectif est de montrer ma manière de travailler :
partir d'un besoin métier, structurer les données, automatiser les contrôles,
produire des indicateurs fiables et aller jusqu'à des recommandations exploitables.

## Pitch 2 minutes

J'ai voulu construire un projet qui ne soit pas simplement un dashboard.

PILOT'IA fonctionne comme une chaîne data complète.

La première étape est la génération de données synthétiques représentant des demandes,
des services, des caisses, des délais de traitement et des ressources.

Ensuite, une couche de contrôle qualité identifie doublons, valeurs manquantes et incohérences.

La partie pilotage calcule les principaux indicateurs : volumes reçus et traités,
backlog, taux de réalisation, taux de réussite, délais médians et respect des SLA.

J'ai ajouté deux briques de data analyse :
une prévision des volumes futurs et une détection d'anomalies.

Enfin, le module charge-ressources permet de tester un scénario :
nombre d'ETP, absentéisme, productivité et évolution attendue de l'activité.
L'application calcule la capacité, le risque de saturation et le besoin éventuel en ETP supplémentaires.

Le projet est conteneurisé avec Docker, testé avec Pytest et prêt pour une CI GitHub.
Il peut fonctionner en démonstration avec des données locales synthétiques ou être raccordé à PostgreSQL.

Le projet illustre donc tout le cycle :
SQL, qualité, analyse, prédiction, restitution et industrialisation.
