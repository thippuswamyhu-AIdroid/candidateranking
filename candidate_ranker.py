from time import time

import_time = time()

from os import cpu_count
from json import loads, load

from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from tqdm import tqdm
from numpy import array, unique, max as np_max, sum as np_sum, mean, exp, log, argmax
from numpy.linalg import norm
from pandas import DataFrame

print(f"\nModules import time: {(time() - import_time):.2f} secs.\n")

class CandidateRankingSystem:

    def __init__(self, candidates_data_file, job_requirements_file, embedding_model_folder, llm_model_file, honeypot_candidates=[]):
        
        start = time()
        
        # Configure Data files
        self.candidates_data_file = candidates_data_file
        self.job_requirements_file = job_requirements_file
        self.honeypot_candidates = honeypot_candidates
        
        # Load Embedding & LLM Model files
        self.embedding_model = SentenceTransformer(embedding_model_folder, device="cpu")
        self.llm_model = Llama(model_path=llm_model_file, n_ctx=512, n_threads=cpu_count(), verbose=False)
        
        print(f"\nInitial Setup Time: {(time() - start):.2f} secs.\n")
        
    def load_candidates_data(self):
        
        """
        Loads the candidates data from the .jsonl file containing 100000 candidate records.
        """
        
        start = time()
        
        self.candidates_data = []

        with open(self.candidates_data_file, "r", encoding="utf-8") as f:
            for line in tqdm(f):
                self.candidates_data.append(loads(line))
                
        print(f"\nLoaded Candidates Data in {(time() - start):.2f} secs.\n")
        
    def load_job_requirements(self):
        
        """
        Loads the job requirements .json file and corresponding requirements defined in the given jod requirements document.
        This file has been pre-generated using a LLM & this step is included within the ranking process.
        """
        
        start = time()
        
        with open(self.job_requirements_file, "r", encoding="utf-8") as f:
            jd_requirements = load(f)
                
        self.ideal_exp_min, self.ideal_exp_max = [int(exp) for exp in jd_requirements["ideal_experience_duration"].split(",")]
        self.exp_min, self.exp_max = [int(exp) for exp in jd_requirements["experience_duration"].split(",")]

        self.desired_experience_areas = [exp_area.strip() for exp_area in jd_requirements["primary_experience_areas"].split(";")]
        self.nice_experience_areas = [exp_area.strip() for exp_area in jd_requirements["secondary_experience_areas"].split(";")]
        self.undesired_experience_areas = [exp_area.strip() for exp_area in jd_requirements["not_preferred_experience"].split(";")]

        self.education_fields = [exp_area.strip() for exp_area in jd_requirements["education_field"].split(";")]
        
        self.desired_skills = jd_requirements["preferred_skills"]
        self.undesired_skills = [exp_area.strip() for exp_area in jd_requirements["not_preferred_skills"].split(";")]

        self.preferred_languages = [lang.strip() for lang in jd_requirements["language"].split(",")]
        
        self.primary_work_location = jd_requirements["primary_work_location"].split(",")
        self.secondary_work_location = jd_requirements["secondary_work_location"].split(",")

        self.preferred_job_titles = jd_requirements["current_job_title"].split(",")
        self.undesired_job_roles = jd_requirements["undesired_roles"].split(",")
        self.preferred_work_industries = jd_requirements["industry"].split(",")

        self.international_candidates_consideration = jd_requirements["international_candidates_consideration"]
        self.international_candidates_relocation_support = jd_requirements["international_relocation_support"]

        self.notice_period_min_days, self.notice_period_max_days = [int(days) for days in jd_requirements["notice_period"].split(",")]
        
        desired_experiences = "\n".join([f"* {experience.capitalize()}" for experience in self.desired_experience_areas])
        desired_education = "\n".join([f"* {education.capitalize()}" for education in self.education_fields])
        desired_skills = "\n".join([f"* {skill['skill_id']} ({skill['proficiency']})" for skill in self.desired_skills["core_required"]])
        desired_job_roles = "\n".join([f"* {role}" for role in self.preferred_job_titles])
        nice_experiences = "\n".join([f"* {experience.capitalize()}" for experience in self.nice_experience_areas])
        undesired_experiences = "\n".join([f"* {experience.capitalize()}" for experience in self.undesired_experience_areas])
        undesired_skills = "\n".join([f"* {skill.capitalize()}" for skill in self.undesired_skills])
        undesired_job_roles = "\n".join([f"* {role}" for role in self.undesired_job_roles])
        
        self.job_description = f"""
Preferred experience areas:
{desired_experiences}
Preferred Education:
{desired_education}
Preferred Skills:
{desired_skills}
Preferred Job Roles (Past or current):
{desired_job_roles}
Nice to have experience areas:
{nice_experiences}
        """
        
        print(f"Loaded Job Requirements in {(time() - start):.2f} secs.\n")
    
    def generate_embeddings(self, texts, batch_size=8, normalize=True, verbose=True):
        
        """
        Build embedding vectors batchwise for the given list of texts.
        """
        
        start = time()

        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=verbose
        )

        embeddings = array([vector.tolist() for vector in embeddings])
        
        print(f"Generated {len(texts)} embedding vectors in {(time() - start):.2f} secs.\n")

        return embeddings
    
    def cosine_similarity(self, query_embs, context_embs):
        
        """
        Compute cosine similarity between 2 arrays of embedding vectors.
        """
    
        dot_products = query_embs @ context_embs.T
        query_embs_norms = norm(query_embs, axis=1, keepdims=True)
        context_embs_norms = norm(context_embs, axis=1, keepdims=True)

        cosine_sims = dot_products / (query_embs_norms @ context_embs_norms.T)

        return cosine_sims
    
    def prepare_embeddings(self, potential_candidates):
        
        """
        Build all the required embeddings across all candidate profile parameters as required for the ranking process.
        """
        
        start = time()
        
        self.candidate_education_fields, self.candidate_skills, self.candidate_certifications, self.candidate_languages = [], [], [], []
        self.candidate_titles, self.candidate_industries, self.candidate_career_descs = [], [], []

        for candidate in tqdm(potential_candidates, desc="Filtering unique entites: "):

            for career in candidate["career_history"]:
                self.candidate_industries.append(career["industry"])
                self.candidate_titles.append(career["title"])
                self.candidate_career_descs.append(career["description"])

            for education in candidate["education"]:
                self.candidate_education_fields.append(f"{education['degree']}, {education['field_of_study']}")

            if candidate["skills"]:
                for skill in candidate["skills"]:
                    self.candidate_skills.append(skill)

            if candidate["certifications"]:
                for certification in candidate["certifications"]:
                    self.candidate_certifications.append(certification)

            if candidate["languages"]:
                for language in candidate["languages"]:
                    self.candidate_languages.append(language)

        print("\nBuilding embeddings on candidate career descriptions:")
        self.candidate_career_descs = sorted(unique(self.candidate_career_descs).tolist())
        self.candidate_career_descs_embs = self.generate_embeddings(texts=self.candidate_career_descs)
        self.candidate_career_descs_embs = {description: self.candidate_career_descs_embs[di] for di, description in enumerate(self.candidate_career_descs)}

        print("Building embeddings on desired experience areas:")
        self.desired_experience_embs = self.generate_embeddings(texts=self.desired_experience_areas)

        print("Building embeddings on nice to have experience areas:")
        self.nice_experience_embs = self.generate_embeddings(texts=self.nice_experience_areas)

        print("Building embeddings on undesired experience areas:")
        self.undesired_experience_embs = self.generate_embeddings(texts=self.undesired_experience_areas)

        print("Building embeddings on preferred industries of work:")
        self.preferred_industry_embs = self.generate_embeddings(texts=self.preferred_work_industries)

        print("Building embeddings on available candidate industries of work:")
        self.candidate_industries = sorted(unique(self.candidate_industries).tolist())
        self.candidate_industries_embs = self.generate_embeddings(texts=self.candidate_industries)
        self.candidate_industries_embs = {industry: self.candidate_industries_embs[ii] for ii, industry in enumerate(self.candidate_industries)}

        print("Building embeddings on preferred job titles:")
        self.preferred_title_embs = self.generate_embeddings(texts=self.preferred_job_titles)

        print("Building embeddings on undesired job titles:")
        self.undesired_title_embs = self.generate_embeddings(texts=self.undesired_job_roles)

        print("Building embeddings on available candidate titles:")
        self.candidate_titles = sorted(unique(self.candidate_titles).tolist())
        self.candidate_title_embs = self.generate_embeddings(texts=self.candidate_titles)
        self.candidate_title_embs = {title: self.candidate_title_embs[ti] for ti, title in enumerate(self.candidate_titles)}

        print("Building embeddings on preferred education degrees:")
        self.preferred_education_embs = self.generate_embeddings(texts=self.education_fields)

        print("Building embeddings on available candidate education degrees:")
        self.candidate_education_fields = sorted(unique(self.candidate_education_fields).tolist())
        self.candidate_education_embs = self.generate_embeddings(texts=self.candidate_education_fields)
        self.candidate_education_embs = {edu_field: self.candidate_education_embs[ei] for ei, edu_field in enumerate(self.candidate_education_fields)}

        print("Building embeddings on preferred primary / required skills:")
        self.preferred_primary_skill_names = sorted(unique([skill["skill_id"] for skill in self.desired_skills["core_required"]]).tolist())
        self.preferred_primary_skill_embs = self.generate_embeddings(texts=self.preferred_primary_skill_names)
        self.preferred_primary_skill_embs = {skill: self.preferred_primary_skill_embs[si] for si, skill in enumerate(self.preferred_primary_skill_names)}
        self.primary_skills_meta = {skill["skill_id"]: skill for skill in self.desired_skills["core_required"]}

        print("Building embeddings on preferred secondary / optional skills:")
        self.preferred_secondary_skill_names = sorted(unique([skill["skill_id"] for skill in self.desired_skills["secondary_skills"]]).tolist())
        self.preferred_secondary_skill_embs = self.generate_embeddings(texts=self.preferred_secondary_skill_names)
        self.preferred_secondary_skill_embs = {skill: self.preferred_secondary_skill_embs[si] for si, skill in enumerate(self.preferred_secondary_skill_names)}
        self.secondary_skills_meta = {skill["skill_id"]: skill for skill in self.desired_skills["secondary_skills"]}

        print("Building embeddings on available candidate skills:")
        self.candidate_skills = sorted(unique([skill["name"] for skill in self.candidate_skills]).tolist())
        self.candidate_skill_embs = self.generate_embeddings(texts=self.candidate_skills)
        self.candidate_skill_embs = {skill: self.candidate_skill_embs[si] for si, skill in enumerate(self.candidate_skills)}

        print("Building embeddings on available candidate certifications:")
        self.candidate_certifications = sorted(unique([cert["name"] for cert in self.candidate_certifications]).tolist())
        self.candidate_cert_embs = self.generate_embeddings(texts=self.candidate_certifications)
        self.candidate_cert_embs = {cert: self.candidate_cert_embs[ci] for ci, cert in enumerate(self.candidate_certifications)}
        
        print(f"Total embeddings preparation time: {(time() - start):.2f} secs.\n")
        
    def profile_score(self, profile):
        
        """
        Compute the candidate profile score based on:
        
        Candidate's:
        * Current title
        * Current industry
        * Years of experience
        
        Preferred (as per JD):
        * Years of experience
        * Preferred current titles
        * Preferred industry
        
        Undesired job titles (as per JD)
        """
    
        current_title, current_industry = profile["current_title"], profile["current_industry"]
        experience_years = profile["years_of_experience"]

        current_title_emb = self.candidate_title_embs[current_title].reshape(1, -1)
        current_industry_emb = self.candidate_industries_embs[current_industry].reshape(1, -1)

        experience_years_score = (experience_years - self.exp_min) / (self.exp_max - self.exp_min)
        current_title_score = np_max(self.cosine_similarity(current_title_emb, self.preferred_title_embs)[0])
        current_industry_score = np_max(self.cosine_similarity(current_industry_emb, self.preferred_industry_embs)[0])
        current_title_irrelevance = 1 - np_max(self.cosine_similarity(current_title_emb, self.undesired_title_embs)[0])

        # Weightage of years of experience > current industry > current title > irrelevance of current title
        score = (0.4 * experience_years_score) + (0.3 * current_industry_score) + (0.2 * current_title_score) + (0.1 * current_title_irrelevance)
        
        return score

    def career_score(self, career):
        
        """
        Compute the candidate career history score based on:
        
        Candidate's:
        * Career role title
        * Career role description
        * Career role industry
        * Career role duration
        * When career role is current or not
        
        &
        
        Preferred (as per JD):
        * Experience areas
        * Job titles
        * Industry
        
        Along with:
        * Nice to have experience areas
        * Undesired experience areas
        * Undesired job titles
        """

        scores = []
        relevant_experience_yrs = 0
        current_exp = []
        relevant_careers = []
        
        def get_weight(param, value):
            # Sets non-linear weights based on candidate's career role duration with a room to provide higher saturating weights when it exceeds the desired requirement
            references = {"duration": 3}
            weight = 1 - exp(-value * log(10) / references[param])
            return weight

        for role in career:

            role_description, role_industry, is_current = role["description"], role["industry"], role["is_current"]
            role_title, role_duration = role["title"], round(role["duration_months"] / 12, ndigits=1)

            description_emb = self.candidate_career_descs_embs[role_description].reshape(1, -1)
            desired_exp_score = np_max(self.cosine_similarity(description_emb, self.desired_experience_embs)[0])
            nice_exp_score = np_max(self.cosine_similarity(description_emb, self.nice_experience_embs)[0])
            undesired_exp_score = 1 - np_max(self.cosine_similarity(description_emb, self.undesired_experience_embs)[0])

            current_title_emb = self.candidate_title_embs[role_title].reshape(1, -1)
            current_title_score = np_max(self.cosine_similarity(current_title_emb, self.preferred_title_embs)[0])
            current_title_irrelevance = 1 - np_max(self.cosine_similarity(current_title_emb, self.undesired_title_embs)[0])

            current_industry_emb = self.candidate_industries_embs[role_industry] .reshape(1, -1)       
            current_industry_score = np_max(self.cosine_similarity(current_industry_emb, self.preferred_industry_embs)[0])

            # Weightage of relevant desired experience areas > relevant nice to have experience areas == industry > title > undesired experience areas
            role_score = (0.4 * desired_exp_score) + (0.2 * nice_exp_score) + (0.2 * current_industry_score) + (0.15 * current_title_score) + (0.05 * undesired_exp_score)
            # Factor in the career role duration
            role_score = (0.7 * role_score) + (0.3 * get_weight("duration", role_duration))
            # Boost career role specific relevance score only if it is a current role and has a relevant current title
            if is_current and (current_title_score > 0.8):
                role_score = min(role_score * 1.1, 1.)
            
            scores.append(role_score), current_exp.append(is_current)
            
            # Accumulate career role details only if has relecant current title, current industry or desired experience for reasoning with LLM later
            if (current_title_score >= 0.8) or (current_industry_score >= 0.75) or (desired_exp_score >= 0.9):
                relevant_careers.append(f"Job role: {role_title} | Industry: {role_industry}")

        return mean(scores) if scores else 0., relevant_careers

    def education_score(self, education, min_relevance=0.8):
        
        """
        Compute candidate education score based on:
        * Candidate's most recent education degree
        * Preferred education degrees
        """

        relevant_educations = []
        
        if education:
            last_end_year = 0
            for edu in education:
                # Score is being calculated purely based on most recent education degree and its relevance
                if edu["end_year"] > last_end_year:
                    education_degree_emb = self.candidate_education_embs[f"{edu['degree']}, {edu['field_of_study']}"].reshape(1, -1)
                    education_score = np_max(self.cosine_similarity(education_degree_emb, self.preferred_education_embs)[0])
                    if education_score >= min_relevance:
                        relevant_educations.append(f"{edu['degree']}, {edu['field_of_study']}")
        else:
            education_score = 0

        return education_score, relevant_educations

    def skills_score(self, skills, min_relevance=0.8):
        
        """
        Compute candidate's skills score based on:
        
        Candidate's:
        * Skill name
        * Skill proficiency
        * Skill duration
        * Skill endorsements
        
        Preferred (as per JD):
        * Primary / Desired / Expected skills and their durations
        * Secondary / Optional / Nice to have skills and their durations
        """

        def get_weight(param, value):
            # Sets non-linear weights based on candidate's skill endorsements, skill duration, required / primary skills coverage and optional / secondary skills coverage with a room to provide higher saturating weights when each exceeds the desired requirement.
            references = {
                "endorsements": 10, "duration": 3, 
                "primary_skill_coverage": len(self.preferred_primary_skill_names),
                "secondary_skill_coverage": len(self.preferred_secondary_skill_names)
            }
            weight = 1 - exp(-value * log(10) / references[param])
            return weight

        relevant_skills = []
        relevant_primary_skills = {skill: 0 for skill in self.preferred_primary_skill_names}
        relevant_secondary_skills = {skill: 0 for skill in self.preferred_secondary_skill_names}

        # Configure weights for skill proficiency
        skill_proficiency_weights = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.75, "expert": 1.}

        for skill in skills:

            name, proficiency = skill["name"], skill["proficiency"]
            duration, endorsements = round(skill["duration_months"] / 12, ndigits=1), skill["endorsements"]

            skill_emb = self.candidate_skill_embs[name].reshape(1, -1)
            primary_skill_scores = self.cosine_similarity(skill_emb, array(list(self.preferred_primary_skill_embs.values())))[0]
            secondary_skill_scores = self.cosine_similarity(skill_emb, array(list(self.preferred_secondary_skill_embs.values())))[0]

            primary_skill_score_idx, secondary_skill_score_idx = argmax(primary_skill_scores), argmax(secondary_skill_scores)

            matched_preferred_primary_skill = self.preferred_primary_skill_names[primary_skill_score_idx]
            matched_primary_skill_score = primary_skill_scores[primary_skill_score_idx]

            matched_preferred_secondary_skill = self.preferred_secondary_skill_names[secondary_skill_score_idx]
            matched_secondary_skill_score = secondary_skill_scores[secondary_skill_score_idx]
            
            if (matched_primary_skill_score >= min_relevance) or (matched_secondary_skill_score >= min_relevance):
                relevant_skills.append(f"{name} ({proficiency})")

            if matched_primary_skill_score >= matched_secondary_skill_score:
                required_skill_duration = self.primary_skills_meta[matched_preferred_primary_skill]["min_months"] / 12
                skill_prominence = (0.2 * get_weight("endorsements", endorsements)) + (0.3 * get_weight("duration", required_skill_duration)) + (0.5 * skill_proficiency_weights[proficiency])
                matched_primary_skill_score = (0.7 * matched_primary_skill_score) + (0.3 * skill_prominence)
                if matched_primary_skill_score > relevant_primary_skills[matched_preferred_primary_skill]:
                    relevant_primary_skills[matched_preferred_primary_skill] = matched_primary_skill_score
            else:
                required_skill_duration = self.secondary_skills_meta[matched_preferred_secondary_skill]["min_months"] / 12
                skill_prominence = (0.2 * get_weight("endorsements", endorsements)) + (0.3 * get_weight("duration", required_skill_duration)) + (0.5 * skill_proficiency_weights[proficiency])
                matched_secondary_skill_score = (0.7 * matched_secondary_skill_score) + (0.3 * skill_prominence)
                if matched_secondary_skill_score > relevant_secondary_skills[matched_preferred_secondary_skill]:
                    relevant_secondary_skills[matched_preferred_secondary_skill] = matched_secondary_skill_score

        existing_primary_skills = [value for value in relevant_primary_skills.values() if value > 0]
        existing_secondary_skills = [value for value in relevant_secondary_skills.values() if value > 0]

        primary_skill_score = (0.55 * mean(existing_primary_skills)) + (0.45 * get_weight("primary_skill_coverage", len(existing_primary_skills))) if existing_primary_skills else 0.
        secondary_skill_score = (0.55 * mean(existing_secondary_skills)) + (0.45 * get_weight("secondary_skill_coverage", len(existing_secondary_skills))) if existing_secondary_skills else 0.
        final_skill_score = (0.7 * primary_skill_score) + (0.3 * secondary_skill_score) 

        return final_skill_score, relevant_skills

    def language_score(self, languages):
        
        """
        Compute candidate's language score based on:
        * Preferred languages and proficiency as per JD
        * Candidate's languages and proficiency
        """

        # Configure weights for language proficiency
        language_weights = {"conversational": 0.5, "professional": 0.75, "native": 1.}

        scores = []
        for language in languages:
            name, proficiency = language["language"], language["proficiency"]
            if name in self.preferred_languages:
                scores.append(language_weights[proficiency])

        return mean(scores) if scores else 0.

    def redrob_signal_score(self, redrob_signals):
        
        """
        Compute candidate's redrob signals score based on the available redrob signals to influence final recommendations
        """

        def get_weight(param, value, proportionality):
            # Sets non-linear weights based on candidate's redrob signals such as applications submitted in last 30 days (higher value indicates candidate is actively looking for a new job role), profile views received in last 30 days (higher value indicates the candidate's activity on redrob resulting into profile views), average response time to messages (lower the better for quick decision making by hiring managers), connection count (higher the better for relevance of the profile and networking), endorsements received (higher the better for authenticity of the expertise) and saved by recruiters in last 30 days (higher the better again indicating the popularity among the hiring managers) with a room to provide higher saturating weights when each exceeds the desired requirement. Only average response time gets higher weight for lower values.
            references = {
                "applications_submitted_30d": 22,
                "profile_views_received_30d": 374,
                "avg_response_time_hours": 279.9,
                "connection_count": 1852,
                "endorsements_received": 239,
                "saved_by_recruiters_30d": 76
            }

            if proportionality == "direct":
                weight = 1 - exp(-value * log(10) / references[param])
            elif proportionality == "inverse":
                weight = exp(-value * log(10) / references[param])
            else:
                weight = 0

            return weight

        unnormalized_signals = []
        normalized_signals = ["profile_completeness_score", "recruiter_response_rate", "github_activity_score", "interview_completion_rate", "offer_acceptance_rate"]

        scores = []
        for signal in normalized_signals:
            score = redrob_signals[signal]
            if score < 0: score = 0
            if signal in ["profile_completeness_score", "github_activity_score"]:
                scores.append(score / 100)
            else:
                scores.append(score)

        for signal in unnormalized_signals:
            proportionality = "inverse" if signal == "avg_response_time_hours" else "direct"
            scores.append(get_weight(signal, redrob_signals[signal], proportionality))

        for additional_signal in ["open_to_work_flag", "verified_email", "verified_phone", "linkedin_connected"]:
            scores.append(int(redrob_signals[additional_signal]))

        return mean(scores)
    
    def run_llm_inference(self, system_prompt, user_prompt, max_tokens=64, temperature=0.15, topP=0.99):
        
        """
        Run LLM model inference for generating reasons to recommend a candidate
        """

        response = self.llm_model.create_chat_completion(
            messages=[
                {"role": "system", "content": "\n".join([system_prompt, "/no_think"])},
                {"role": "user", "content": "\n".join([user_prompt, "/no_think"])}
            ],
            max_tokens=max_tokens, temperature=temperature, top_p=topP
        )
        response_text = response["choices"][0]["message"]["content"]
        
        return response_text
    
    def generate_reasons(self, candidate_rank, profile_summary, total_experience, relevant_educations, relevant_careers, relevant_skills):
        
        """
        Generate reason for recommending a candidate based on the candidate's:
        * Relevant career histories if available
        * Relevant education degrees if available
        * Relevant skills if available
        * Profile summary if none of the above are available
        """
        
        career_description = "; ".join([career for career in relevant_careers])
        education = "; ".join([degree for degree in relevant_educations[:2]])
        skills = ", ".join([skill for skill in relevant_skills])
        
        career_profile_details = ""
        if relevant_careers:
            career_profile_details += f"\n\nCareer Details:\n{career_description}"
        if relevant_educations:
            career_profile_details += f"\n\nEducation:\n{education}"
        if relevant_skills:
            career_profile_details += f"\n\nSkills Set:\n{skills}"
        if not career_profile_details:
            career_profile_details = f"Profile summary:\n{profile_summary}\n" + career_profile_details
        career_profile_details = f"Total experience: {total_experience} years.\n" + career_profile_details
        
        system_prompt = f"""
/no_think
Given a candidate profile details for a Senior AI Engineer role, provide a valid reason based on only these details in single short sentence of max 15 words to consider this profile for the job role.
/no_think
"""
        
        user_prompt = f"""
/no_think
Here is a candidate profile ranked {candidate_rank} within top 100 candidates list:
```
{career_profile_details}
```
Generate a reason to consider this profile using < 15 words in plain text.
/no_think
"""
            
        ranking_reason = self.run_llm_inference(system_prompt, user_prompt)
        
        separator = "</think>"
        ranking_reason = ranking_reason[ranking_reason.find(separator)+len(separator):]
        
        return ranking_reason.strip().capitalize()        
    
    def rank_candidates(self, topK=100):
        
        """
        1. Perform candidate profile relevance check for the job role.
        2. Rank them based on the various profile section specific scores.
        3. Generate reasons for recommending a candidate only for top 100 (configurable) candidates.
        4. Save the recommendations into a data frame.
        """
        
        main_start = time()
        start = main_start
        
        self.load_candidates_data()
        self.load_job_requirements()
        
        yoe_criteria = lambda x: (self.exp_min <= x <= self.exp_max)
        location_criteria = lambda country, city: (city in (self.primary_work_location + self.secondary_work_location)) or ((((country == "India") and (city not in (self.primary_work_location + self.secondary_work_location))) or (self.international_candidates_consideration and (country != "india"))) and data["redrob_signals"]["willing_to_relocate"])
        notice_period_criteria = lambda x: self.notice_period_min_days <= x <= self.notice_period_max_days
        work_mode_criteria = lambda x: (x != "remote")

        potential_candidates = []
        for data in tqdm(self.candidates_data, desc="Candidates initial filtering using Experience, Location, Notice Period & Preferred Work Mode: "):
            if yoe_criteria(data["profile"]["years_of_experience"]) and \
                location_criteria(data["profile"]["country"].capitalize(), data["profile"]["location"].capitalize()) and \
                notice_period_criteria(data["redrob_signals"]["notice_period_days"]) and \
                work_mode_criteria(data["redrob_signals"]["preferred_work_mode"]):
                    potential_candidates.append(data)

        print(f"\nOriginal Candidates Count: {len(self.candidates_data)} | Potential Candidates Count: {len(potential_candidates)}\n")
        
        self.prepare_embeddings(potential_candidates)
        
        scored_candidates = []
        
        for candidate in tqdm(potential_candidates, desc="Ranking Potential Candidates: "):

            candidate_id = candidate["candidate_id"]
                                    
            # Filter out honeypot candidates if provided
            if self.honeypot_candidates and (candidate_id in self.honeypot_candidates):
                continue
                                    
            profile, career, education = candidate["profile"], candidate["career_history"], candidate["education"]
            skills, certifications, languages = candidate["skills"], candidate["certifications"], candidate["languages"]
            redrob_signals = candidate["redrob_signals"]

            pscore = self.profile_score(profile)
            lscore = self.language_score(languages)
            rscore = self.redrob_signal_score(redrob_signals)
            
            escore, relevant_educations = self.education_score(education)
            cscore, relevant_careers = self.career_score(career)
            sscore, relevant_skills = self.skills_score(skills)
            
            candidate_score = (0.25 * pscore) + (0.4 * cscore) + (0.15 * sscore) + (0.1 * escore) + (0.05 * lscore) + (0.05 * rscore)

            scored_candidates.append([candidate_id, profile, relevant_educations, relevant_careers, relevant_skills, candidate_score, pscore, cscore, sscore, escore, rscore, lscore])
            
        print()
        scored_candidates = sorted(scored_candidates, key=lambda x: float(x[-7]), reverse=True)[:topK]
        for ci, candidate in enumerate(tqdm(scored_candidates, desc="Generating Ranking Reasons: ")):
            candidate_id, profile, relevant_educations, relevant_careers, relevant_skills, candidate_score, pscore, cscore, sscore, escore, rscore, lscore = candidate
            ranking_reason = self.generate_reasons(ci+1, profile["summary"], profile["years_of_experience"], relevant_educations, relevant_careers, relevant_skills)
            scored_candidates[ci] = [candidate_id, ci+1, candidate_score, ranking_reason]

        scored_candidates_headers = ["candidate_id", "rank", "score", "reasoning"]
        scored_candidates_df = DataFrame(scored_candidates, columns=scored_candidates_headers)
        
        out_file_name = "team_AIStriversBMSCE"
        scored_candidates_df.to_csv(f"./data/{out_file_name}.csv", index=False, encoding="utf-8")
        scored_candidates_df.to_excel(f"./data/{out_file_name}.xlsx", index=False)
        
        print(f"\nTotal candidates ranking time: {(time() - main_start):.2f} secs.\n")
        
if __name__ == "__main__":
    
    honeypot_candidates = [
        'CAND_0003430', 'CAND_0003582', 'CAND_0005291', 'CAND_0007413', 'CAND_0008978', 'CAND_0010770', 'CAND_0011125', 
        'CAND_0012837', 'CAND_0013536', 'CAND_0016000', 'CAND_0016678', 'CAND_0024752', 'CAND_0025579', 'CAND_0030946', 
        'CAND_0032996', 'CAND_0033131', 'CAND_0033817', 'CAND_0033972', 'CAND_0036299', 'CAND_0036839', 'CAND_0038431', 
        'CAND_0039754', 'CAND_0040955', 'CAND_0042245', 'CAND_0044252', 'CAND_0046649', 'CAND_0046689', 'CAND_0048740', 
        'CAND_0050553', 'CAND_0050876', 'CAND_0052478', 'CAND_0053527', 'CAND_0055792', 'CAND_0055992', 'CAND_0056983', 
        'CAND_0057529', 'CAND_0060072', 'CAND_0060642', 'CAND_0061265', 'CAND_0061722', 'CAND_0063888', 'CAND_0064256', 
        'CAND_0065096', 'CAND_0066405', 'CAND_0067443', 'CAND_0067535', 'CAND_0070429', 'CAND_0071115', 'CAND_0072379', 
        'CAND_0073504', 'CAND_0073853', 'CAND_0074119', 'CAND_0074735', 'CAND_0077250', 'CAND_0078042', 'CAND_0080102', 
        'CAND_0080291', 'CAND_0086808', 'CAND_0088354', 'CAND_0090900', 'CAND_0091068', 'CAND_0091534', 'CAND_0093331', 
        'CAND_0094482', 'CAND_0095140', 'CAND_0095317', 'CAND_0095480', 'CAND_0095619', 'CAND_0096150', 'CAND_0098288'
    ]
    
    candidates_ranker = CandidateRankingSystem(
        candidates_data_file="./data/candidates.jsonl",
        job_requirements_file="./job_requirements.json",
        embedding_model_folder="./models/BGE-embed/BAAI/bge-small-en-v1.5",
        llm_model_file="./models/Qwen-GGUF/Qwen3-0.6B-Q4_K_M.gguf",
        honeypot_candidates=honeypot_candidates
    )
    
    candidates_ranker.rank_candidates(topK=100)
    
    print(f"End to end execution time: {(time() - import_time):.2f} secs.\n")
    