FlyBot est un chatbot intelligent dédié au domaine du voyage, permettant de répondre aux questions des utilisateurs sur les vols.
Le projet combine une interface web conviviale avec une architecture backend basée sur FastAPI, LangChain, ChromaDB, Gemini AI et des embeddings HuggingFace.
FlyBot est connecté à une base locale de données de vols (CSV), et propose des réponses personnalisées sur :

✈️ horaires de vol
🛄 politique de bagages
🍴 services à bord et repas
❌ annulations & retards
💰 promotions et offres spéciales
📍 suggestions basées sur la localisation
🎟️ réservation de vols (simulation)

Le projet comprend :

un frontend HTML animé avec splash screen

un backend API REST

un moteur de recherche vectoriel pour le question/réponse

une intégration de LLM (Gemini) pour des réponses naturelles
