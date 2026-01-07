"""
Sample Drug Data for PharmaBula MVP

Since ANVISA's API requires authentication and blocks automated access,
this module provides sample drug bulletin data for demonstration purposes.

In production, this data would come from:
1. Manual data entry by pharmacists
2. Partnership with ANVISA for API access
3. Licensed pharmaceutical databases
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.scrapers.anvisa_scraper import DrugBulletin


# Sample drug data for common medications in Brazil
SAMPLE_DRUGS: list[DrugBulletin] = [
    DrugBulletin(
        id="paracetamol_001",
        name="Paracetamol 500mg",
        company="Laboratório Genérico",
        active_ingredient="Paracetamol",
        bulletin_type="paciente",
        pdf_url=None,
        text_content="""
PARACETAMOL 500mg - BULA DO PACIENTE

INDICAÇÕES
O paracetamol é indicado para o alívio temporário de dores leves a moderadas, 
como dor de cabeça, dor muscular, dor de dente, dor nas costas, cólicas menstruais, 
e para redução da febre.

POSOLOGIA
Adultos e crianças acima de 12 anos: Tomar 1 a 2 comprimidos a cada 4 a 6 horas, 
não excedendo 8 comprimidos em 24 horas.
Crianças de 6 a 12 anos: Tomar 1/2 a 1 comprimido a cada 4 a 6 horas.

CONTRAINDICAÇÕES
- Hipersensibilidade ao paracetamol ou a qualquer componente da fórmula
- Doença hepática grave
- Uso concomitante de outros medicamentos contendo paracetamol

EFEITOS COLATERAIS
Reações raras: Reações alérgicas (erupção cutânea, urticária)
Reações muito raras: Alterações nos exames de sangue, reações hepáticas

INTERAÇÕES MEDICAMENTOSAS
- Álcool: aumenta o risco de danos ao fígado
- Varfarina: pode aumentar o efeito anticoagulante
- Medicamentos para epilepsia: podem alterar a eficácia do paracetamol

ADVERTÊNCIAS
- Não exceda a dose recomendada
- Em caso de ingestão acidental de doses maiores, procure atendimento médico
- Não use por mais de 3 dias para febre ou 10 dias para dor sem orientação médica
- Pacientes com problemas hepáticos devem consultar um médico antes de usar
        """,
        last_updated=datetime.now()
    ),
    DrugBulletin(
        id="dipirona_001",
        name="Dipirona Sódica 500mg",
        company="Laboratório Nacional",
        active_ingredient="Dipirona Sódica (Metamizol)",
        bulletin_type="paciente",
        pdf_url=None,
        text_content="""
DIPIRONA SÓDICA 500mg - BULA DO PACIENTE

INDICAÇÕES
A dipirona é indicada como analgésico e antipirético para:
- Dor de cabeça
- Dor de dente
- Dor pós-operatória
- Cólicas
- Febre

POSOLOGIA
Adultos e adolescentes acima de 15 anos: 1 a 2 comprimidos até 4 vezes ao dia.
Dose máxima diária: 4g (8 comprimidos).
Crianças: Consultar médico para dose adequada ao peso.

CONTRAINDICAÇÕES
- Alergia à dipirona ou a outros derivados pirazolônicos
- História de reações alérgicas graves a analgésicos
- Função da medula óssea comprometida
- Deficiência de G6PD
- Gravidez (especialmente primeiro e terceiro trimestres)

EFEITOS COLATERAIS
Reações incomuns: Reações alérgicas na pele
Reações raras: Agranulocitose (redução grave dos glóbulos brancos)
Reações muito raras: Choque anafilático, síndrome de Stevens-Johnson

INTERAÇÕES MEDICAMENTOSAS
- Ciclosporina: pode reduzir níveis sanguíneos
- Metotrexato: pode aumentar toxicidade
- Anticoagulantes orais: pode potencializar efeito

ADVERTÊNCIAS
- Não use se tiver histórico de reações graves a analgésicos
- Suspenda o uso e procure médico se apresentar febre, dor de garganta ou lesões na boca
- Evite uso prolongado sem orientação médica
        """,
        last_updated=datetime.now()
    ),
    DrugBulletin(
        id="ibuprofeno_001",
        name="Ibuprofeno 400mg",
        company="Pharma Brasil",
        active_ingredient="Ibuprofeno",
        bulletin_type="paciente",
        pdf_url=None,
        text_content="""
IBUPROFENO 400mg - BULA DO PACIENTE

INDICAÇÕES
O ibuprofeno é um anti-inflamatório não esteroidal (AINE) indicado para:
- Dores leves a moderadas
- Dor de cabeça e enxaqueca
- Dor muscular e nas articulações
- Dor de dente
- Cólicas menstruais
- Febre
- Inflamações

POSOLOGIA
Adultos e crianças acima de 12 anos: 200mg a 400mg a cada 4 a 6 horas.
Dose máxima diária: 1200mg (3 comprimidos de 400mg).
Tomar preferencialmente com alimentos para reduzir irritação gástrica.

CONTRAINDICAÇÕES
- Alergia ao ibuprofeno ou outros AINEs
- Úlcera péptica ativa ou sangramento gastrointestinal
- Insuficiência cardíaca grave
- Insuficiência renal ou hepática grave
- Último trimestre da gravidez
- Histórico de asma induzida por AINEs

EFEITOS COLATERAIS
Comuns: Dor de estômago, náuseas, diarreia, gases
Incomuns: Dor de cabeça, tontura, retenção de líquidos
Raros: Úlcera gástrica, sangramento, reações alérgicas

INTERAÇÕES MEDICAMENTOSAS
- Aspirina: reduz efeito cardioprotetor
- Anticoagulantes: aumenta risco de sangramento
- Anti-hipertensivos: pode reduzir eficácia
- Lítio: aumenta níveis no sangue
- Metotrexato: aumenta toxicidade

ADVERTÊNCIAS
- Tomar com alimentos ou leite
- Não usar por períodos prolongados sem orientação médica
- Idosos têm maior risco de efeitos colaterais gastrointestinais
- Evitar em pacientes com problemas cardíacos
        """,
        last_updated=datetime.now()
    ),
    DrugBulletin(
        id="omeprazol_001",
        name="Omeprazol 20mg",
        company="MedPharma",
        active_ingredient="Omeprazol",
        bulletin_type="paciente",
        pdf_url=None,
        text_content="""
OMEPRAZOL 20mg - BULA DO PACIENTE

INDICAÇÕES
O omeprazol é um inibidor da bomba de prótons indicado para:
- Úlcera gástrica e duodenal
- Doença do refluxo gastroesofágico (azia)
- Síndrome de Zollinger-Ellison
- Prevenção de úlceras causadas por anti-inflamatórios
- Erradicação do H. pylori (em combinação com antibióticos)

POSOLOGIA
Adultos:
- Úlcera duodenal: 20mg uma vez ao dia por 2 a 4 semanas
- Úlcera gástrica: 20mg uma vez ao dia por 4 a 8 semanas
- Refluxo: 20mg uma vez ao dia por 4 a 8 semanas
Tomar em jejum, 30 minutos antes do café da manhã.
Engolir o comprimido inteiro, não mastigar ou triturar.

CONTRAINDICAÇÕES
- Alergia ao omeprazol ou benzimidazóis
- Uso concomitante com nelfinavir (medicamento para HIV)

EFEITOS COLATERAIS
Comuns: Dor de cabeça, diarreia, dor abdominal, náuseas
Incomuns: Tontura, constipação, flatulência
Raros: Alterações nas enzimas hepáticas, reações alérgicas

INTERAÇÕES MEDICAMENTOSAS
- Clopidogrel: pode reduzir eficácia
- Metotrexato: pode aumentar níveis
- Antifúngicos (cetoconazol): absorção reduzida
- Diazepam: metabolismo alterado
- Digoxina: absorção aumentada

ADVERTÊNCIAS
- Uso prolongado pode causar deficiência de vitamina B12 e magnésio
- Pode aumentar risco de fraturas ósseas com uso prolongado
- Antes de iniciar, excluir possibilidade de câncer gástrico
        """,
        last_updated=datetime.now()
    ),
    DrugBulletin(
        id="losartana_001",
        name="Losartana Potássica 50mg",
        company="CardioFarma",
        active_ingredient="Losartana Potássica",
        bulletin_type="paciente",
        pdf_url=None,
        text_content="""
LOSARTANA POTÁSSICA 50mg - BULA DO PACIENTE

INDICAÇÕES
A losartana é um bloqueador do receptor de angiotensina II indicado para:
- Hipertensão arterial (pressão alta)
- Proteção renal em pacientes diabéticos tipo 2 com proteinúria
- Insuficiência cardíaca quando IECAs não são tolerados
- Redução do risco de AVC em pacientes hipertensos com hipertrofia ventricular

POSOLOGIA
Hipertensão:
- Dose inicial: 50mg uma vez ao dia
- Dose máxima: 100mg uma vez ao dia
- Pode ser tomado com ou sem alimentos
Insuficiência cardíaca:
- Dose inicial: 12,5mg uma vez ao dia
- Aumentar gradualmente conforme tolerância

CONTRAINDICAÇÕES
- Alergia à losartana ou componentes da fórmula
- Gravidez (pode causar danos ao feto)
- Amamentação
- Uso concomitante com alisquireno em diabéticos

EFEITOS COLATERAIS
Comuns: Tontura, infecções respiratórias superiores
Incomuns: Hipotensão, aumento de potássio no sangue
Raros: Angioedema (inchaço da face e garganta)

INTERAÇÕES MEDICAMENTOSAS
- Diuréticos poupadores de potássio: risco de hipercalemia
- AINEs (ibuprofeno): podem reduzir efeito anti-hipertensivo
- Lítio: aumento dos níveis sanguíneos
- Suplementos de potássio: evitar uso conjunto

ADVERTÊNCIAS
- Não usar durante a gravidez
- Monitorar função renal e potássio periodicamente
- Pode causar tontura; cuidado ao dirigir
- Manter hidratação adequada
        """,
        last_updated=datetime.now()
    ),
    DrugBulletin(
        id="amoxicilina_001",
        name="Amoxicilina 500mg",
        company="Antibióticos Brasil",
        active_ingredient="Amoxicilina Triidratada",
        bulletin_type="paciente",
        pdf_url=None,
        text_content="""
AMOXICILINA 500mg - BULA DO PACIENTE

INDICAÇÕES
A amoxicilina é um antibiótico da classe das penicilinas indicado para:
- Infecções do trato respiratório (sinusite, otite, amigdalite, bronquite)
- Infecções urinárias
- Infecções de pele
- Erradicação do H. pylori (em combinação com outros medicamentos)
- Prevenção de endocardite bacteriana

POSOLOGIA
Adultos e crianças acima de 40kg:
- Infecções leves a moderadas: 500mg a cada 8 horas
- Infecções graves: 500mg a cada 8 horas ou 875mg a cada 12 horas
- Duração: 7 a 14 dias conforme infecção
Crianças: Dose calculada pelo peso (20-40mg/kg/dia divididos em 3 doses)

CONTRAINDICAÇÕES
- Alergia a penicilinas ou cefalosporinas
- Mononucleose infecciosa (alto risco de erupção cutânea)

EFEITOS COLATERAIS
Comuns: Diarreia, náuseas, erupção cutânea
Incomuns: Vômitos, candidíase oral ou vaginal
Raros: Reações alérgicas graves, colite pseudomembranosa

INTERAÇÕES MEDICAMENTOSAS
- Metotrexato: toxicidade aumentada
- Anticoagulantes orais: pode aumentar efeito
- Contraceptivos orais: eficácia pode ser reduzida
- Probenecida: aumenta níveis de amoxicilina

ADVERTÊNCIAS
- Completar todo o tratamento prescrito
- Informar ao médico sobre alergias a antibióticos
- Em caso de diarreia grave, procurar atendimento médico
- Pode causar reações em pessoas alérgicas à penicilina
        """,
        last_updated=datetime.now()
    ),
    DrugBulletin(
        id="metformina_001",
        name="Metformina 850mg",
        company="DiabetesCare",
        active_ingredient="Cloridrato de Metformina",
        bulletin_type="paciente",
        pdf_url=None,
        text_content="""
METFORMINA 850mg - BULA DO PACIENTE

INDICAÇÕES
A metformina é um antidiabético oral indicado para:
- Diabetes mellitus tipo 2, especialmente em pacientes com sobrepeso
- Pré-diabetes (em alguns casos)
- Síndrome dos ovários policísticos (uso off-label)

POSOLOGIA
Dose inicial: 500mg ou 850mg uma vez ao dia com as refeições
Dose de manutenção: 850mg a 1000mg duas a três vezes ao dia
Dose máxima: 2550mg por dia divididos em doses
Aumentar dose gradualmente para reduzir efeitos gastrointestinais

CONTRAINDICAÇÕES
- Insuficiência renal grave
- Acidose metabólica, incluindo cetoacidose diabética
- Insuficiência hepática grave
- Insuficiência cardíaca descompensada
- Consumo excessivo de álcool
- Desidratação grave

EFEITOS COLATERAIS
Muito comuns: Náuseas, vômitos, diarreia, dor abdominal, perda de apetite
Comuns: Alteração no paladar (gosto metálico)
Raros: Acidose láctica (condição grave), deficiência de vitamina B12

INTERAÇÕES MEDICAMENTOSAS
- Álcool: aumenta risco de acidose láctica
- Contrastes iodados: suspender metformina antes e após exames
- Diuréticos: podem afetar função renal
- Corticoides: podem aumentar glicemia

ADVERTÊNCIAS
- Tomar durante ou após as refeições
- Suspender antes de cirurgias ou exames com contraste
- Monitorar função renal periodicamente
- Sintomas de acidose láctica: náuseas intensas, dor muscular, fadiga extrema
        """,
        last_updated=datetime.now()
    ),
    DrugBulletin(
        id="sinvastatina_001",
        name="Sinvastatina 20mg",
        company="CardioMed",
        active_ingredient="Sinvastatina",
        bulletin_type="paciente",
        pdf_url=None,
        text_content="""
SINVASTATINA 20mg - BULA DO PACIENTE

INDICAÇÕES
A sinvastatina é uma estatina indicada para:
- Redução do colesterol LDL (colesterol ruim)
- Aumento do colesterol HDL (colesterol bom)
- Redução de triglicerídeos
- Prevenção de eventos cardiovasculares
- Doença arterial coronariana

POSOLOGIA
Dose inicial: 10mg a 20mg uma vez ao dia à noite
Dose máxima: 40mg por dia (80mg apenas em casos específicos)
Tomar preferencialmente à noite, pois a produção de colesterol é maior durante o sono

CONTRAINDICAÇÕES
- Doença hepática ativa
- Gravidez e amamentação
- Uso concomitante com gemfibrozila, ciclosporina, danazol
- Alergia à sinvastatina

EFEITOS COLATERAIS
Comuns: Dor de cabeça, constipação, náuseas, dor abdominal
Incomuns: Dores musculares, fraqueza, cãibras
Raros: Rabdomiólise (destruição muscular grave), alterações hepáticas

INTERAÇÕES MEDICAMENTOSAS
- Amiodarona, verapamil, diltiazem: aumentam risco de miopatia
- Varfarina: pode aumentar efeito anticoagulante
- Suco de toranja (grapefruit): evitar consumo
- Antibióticos macrolídeos: aumentam níveis de sinvastatina

ADVERTÊNCIAS
- Relatar imediatamente dores musculares inexplicadas
- Fazer exames de função hepática periodicamente
- Não usar durante gravidez ou se planeja engravidar
- Evitar consumo excessivo de álcool
        """,
        last_updated=datetime.now()
    ),
]


def get_sample_drugs() -> list[DrugBulletin]:
    """Return list of sample drug bulletins for demo purposes."""
    return SAMPLE_DRUGS


def get_drug_by_name(name: str) -> Optional[DrugBulletin]:
    """Find a drug by name (case insensitive partial match)."""
    name_lower = name.lower()
    for drug in SAMPLE_DRUGS:
        if name_lower in drug.name.lower() or name_lower in drug.active_ingredient.lower():
            return drug
    return None
