# Prefiltering: map same kinases with different names and remove non-vertebrate proteins (also reduce some rare subvariants to the more general category)
# Sadly, there's just no way around a lot of manual mapping
prefiltering_mapping = {
    "GRK-1": "GRK1",
    "GRK-2": "GRK2",
    "GRK-3": "GRK3",
    "GRK-4": "GRK4",
    "GRK-5": "GRK5",
    "GRK-6": "GRK6",
    "AlphaK3": "MAK",
    "AURKA": "AurA",
    "AURKB": "AurB",
    "AURKC": "AurC",
    "Aurora A": "AurA",
    "Aurora B": "AurB",
    "BRSK1 iso2": "BRSK1",
    "ABL2": "Arg",
    "Abl2": "Arg",
    "ABL1": "Abl",
    "BCR-ABL1": "Abl",
    "ACVR1B": "ALK4",
    "ATG1": "ULK1",
    "Atg1": "ULK1",
    "Axl": "AXL",
    "Bcr": "Abl",
    "ABL": "Abl",
    "Bcr": "Abl",
    "BCR-ABL1": "Abl",
    "Brk": "BRK",
    "Btk": "BTK",
    "BARK": "GRK2",
    "BARK1": "GRK2",
    "CAMK1A": "CAMK1",
    "CAMK1B": "CAMK1",
    "CAMK1G": "CAMK1",
    "CAMK2A": "CAMK2",
    "CAMK2B": "CAMK2",
    "CAMK2D": "CAMK2",
    "CAMK2D iso8": "CAMK2",
    "CAMK2G": "CAMK2",
    "CAMKK1": "CAMK",
    "CAMKK2": "CAMK",
    "CCDPK": "CAMK",
    "CDK11A": "CDK11",
    "CDK11A iso10": "CDK11",
    "CDK11B": "CDK11",
    "CDKL5": "STK9",
    "CDKL2": "CDKL2",
    "CDPK": "CK",
    "CHEK1": "CHK1",
    "CHEK2": "CHK2",
    "CIT": "CRIK",
    "CK1A": "CK1",
    "CK1A2": "CK1",
    "CK1D": "CK1",
    "CK1E": "CK1",
    "CK1G1": "CK1",
    "CK1G2": "CK1",
    "CK1G3": "CK1",
    "CK1": "CK1",
    "CK1_alpha": "CK1",
    "CkI_alpha": "CK1",
    "CK1_delta": "CK1",
    "CK1_epsilon": "CK1",
    "CkII": "CK2",
    "CkII_beta": "CK2",
    "CK2A2": "CK2",
    "CK2B": "CK2",
    "CK2_alpha": "CK2",
    "CK2_beta": "CK2",
    "CLA4": "CLA4",
    "CPK": "CK",
    "CPK1": "CK1",
    "CK3": "CK3",
    "CPK32": "CK",
    "CPK34": "CK",
    "CSF1R": "FMS",
    "CSFR": "FMS",
    "CSNK1D": "CK1",
    "CSNK1E": "CK1",
    "CaM-KI": "CAMK1",
    "CaM-KI_alpha": "CAMK1",
    "CaM-KII": "CAMK2",
    "CaM-KII_alpha": "CAMK2",
    "CaM-KII_delta": "CAMK2",
    "CaM-KIV": "CAMK4",
    "CaM-KK_alpha": "CaMKK1",
    "Caki": "CAMK",
    "CamK2": "CAMK2",
    "Cdk1": "CDK1",
    "ChaK1": "TRPM7",
    "ChaK2": "TRPM7",
    "Cot": "MAP3K8",
    "DNA-PK": "DNAPK",
    "DUN1": "CHK2",
    "EEF2K": "CAMK2",
    "EIF2AK2": "PKR",
    "EIF2AK3": "PKR",
    "Eg3 kinase": "MELK",
    "FAK iso2": "FAK",
    "FAK1": "FAK",
    "FAK2": "FAK",
    "Fer": "FER",
    "Fes": "FES",
    "FYN": "Fyn",
    "GSK-3_alpha": "GSK3A",
    "GSK-3_beta": "GSK3B",
    "GSK3B iso2": "GSK3B",
    "GWL": "MASTL",
    "Haspin": "HASPIN",
    "IKK_alpha": "IKKA",
    "IKK_beta": "IKKB",
    "IKK_epsilon": "IKKE",
    "ILK1": "ILK",
    "JNK1 iso2": "JNK1",
    "JNK2 iso2": "JNK2",
    "Lck": "LCK",
    "Lyn": "LYN",
    "MATK": "CHK",
    "MARK3 iso3": "MARK3",
    "MEK1": "MAP2K1",
    "MEK2": "MAP2K2",
    "MEK5": "MAP2K5",
    "MEKK1": "MAP3K1",
    "MEKK2": "MAP3K2",
    "MEKK3": "MAP3K3",
    "MEKK4": "MAP3K4",
    "MEKK6": "MAP3K6",
    "MKK3": "MAP2K3",
    "MKK4": "MAP2K4",
    "MKK5": "MAP2K5",
    "MKK6": "MAP2K6",
    "MKK7": "MAP2K7",
    "MKNK2": "GPRK7",
    "MPK3": "MAPK3",
    "MPK6": "MAPK6",
    "MRCKa": "MRCKA",
    "MTOR": "mTOR",
    "Met": "MET",
    "Mlck": "MLCK",
    "Mnk1 iso2": "Mnk",
    "NPM-ALK": "ALK",
    "NTRK2": "TRKB",
    "NuaK1": "NUAK1",
    "NuaK2": "NUAK2",
    "OXSR1": "OSR1",
    "PDGFR_alpha": "PDGFRA",
    "PDGFR_beta": "PDGFRB",
    "PDHK1": "PDK1",
    "PDHK2": "PDK2",
    "PDHK3": "PDK3",
    "PDHK4": "PDK4",
    "PDK-1": "PDK1",
    "PDK-2": "PDK2",
    "PDPK1": "PDK1",
    "PERK": "PKR",
    "PHKA1": "PHK",
    "PHKB": "PHK",
    "PKC_iota": "PKC",
    "PHKG1": "PHK",
    "PHKG2": "PHK",
    "PIM-1": "PIM1",
    "skMLCK": "MLCK",
    "smMLCK": "MLCK",
    "p70S6K iso2": "p70S6K",
    "p90RSK": "RSK",
    "p58": "CDK11",
    "mbk": "DYRK1A",
    "mbk-2": "DYRK",
    "mapk": "MAPK",
    "ksg1": "PDK1",
    "atr": "ATR",
    "atm": "ATM",
    "cdk": "CDK",
    "cdk1": "CDK1",
    "cdk2": "CDK2",
    "cdc2": "CDC2",
    "cak": "CAK",
    "caMLCK": "MLCK",
    "aurka": "AurA",
    "ZIPK/DAPK3": "DAPK3",
    "Yes": "YES",
    "wee1": "Wee1",
    "Wee2": "WEE",
    "WEE1": "Wee1",
    "WEE1B": "Wee1b",
    "VRK2 iso2": "VRK2",
    "VEGFR1": "VEGFR",
    "VEGFR2": "VEGFR",
    "VEGFR3": "VEGFR",
    "UHMK1": "KIS",
    "Uhmk1": "KIS",
    "Tyk2": "TYK2",
    "Tyr": "TK",
    "TrkA": "TRKA",
    "TrkB": "TRKB",
    "TrkC": "TRKC",
    "Tao": "TAO",
    "TRB2": "Trb2",
    "Tor": "TOR",
    "mTOR": "TOR",
    "TOR2": "TOR",
    "TORC1": "TOR",
    "TORC2": "TOR",
    "TGFBR1": "TGFbR1",
    "TGFBR2": "TGFbR2",
    "TGFBR2 iso1": "TGFbR2",
    "TAOK1": "TAO1",
    "Syk": "SYK",
    "Src": "SRC",
    "Src iso1": "SRC",
    "STK10": "LOK",
    "STK24/MST3": "MST3",
    "STK3/MST2": "MST2",
    "STK4/MST1": "MST1",
    "SRMS": "SRM",
    "SIK1": "SIK",
    "SIK2": "SIK",
    "SGK1": "SGK",
    "Ron": "RON",
    "Ret": "RET",
    "Ret iso3": "RET",
    "RSK-1": "RSK1",
    "RSK-2": "RSK2",
    "RSK-5": "RSK5",
    "RPS6KA1": "RSK1",
    "RPS6KA3": "RSK2",
    "RPS6KA4": "MSK2",
    "RPS6KA5": "MSK1",
    "RPS6KB1": "p70S6K",
    "Pink1": "PINK1",
    "Pim1": "PIM1",
    "Pim2": "PIM2",
    "Pim3": "PIM3",
    "Pak": "PAKA",
    "PAK": "PAKA",
    "PTK6": "BRK",
    "PRKD2": "PKD2",
    "PRKD1": "PKD1",
    "PRKD3": "PKD3",
    "PKM iso2": "PKM",
    "PKMYT1": "MYT1",
    "PKG1/cGK-I": "PKG1",
    "PKG2/cGK-II": "PKG2",
    "PKG/PRKG1": "PKG1",
    "PKG/PRKG2": "PKG2",
    "PKG/cGK": "PKG",
    "PKG1 iso2": "PKG1",
    "PKD/PRKD1": "PKD1",
    "PKD/PRKD2": "PKD2",
    "PKDCC": "SgK493",
    "PKC_alpha": "PKCa",
    "PKC_beta": "PKCb",
    "PKC_delta": "PKCd",
    "PKC_epsilon": "PKCe",
    "PKC_eta 2": "PKCh",
    "PKC_gamma": "PKCg",
    "PKC_theta": "PKCt",
    "PKC_zeta": "PKCz",
    "PKCA": "PKCa",
    "PKCB": "PKCb",
    "PKCB iso2": "PKCb",
    "PKCD": "PKCd",
    "PKCE": "PKCe",
    "PKCG": "PKCg",
    "PKCH": "PKCh",
    "PKCI": "PKCi",
    "PKCT": "PKCt",
    "PKCZ": "PKCz",
    "PKC/PRKCA": "PKCa",
    "PKC/PRKCB": "PKCb",
    "PKC/PRKCD": "PKCd",
    "PKC/PRKCE": "PKCe",
    "PKC/PRKCG": "PKCg",
    "PKC/PRKCH": "PKCh",
    "PKC/PRKCI": "PKCi",
    "PKC/PRKCQ": "PKCt",
    "PKC/PRKCZ": "PKCz",
    "PKB/AKT1": "Akt1",
    "PKB/AKT2": "Akt2",
    "PKB/Akt1": "Akt1",
    "PKB_beta": "PKB",
    "PKA_alpha": "PKACa",
    "PKACA": "PKACa",
    "PKACA iso2": "PKACa",
    "PKACB": "PKACb",
    "plk": "PLK",
    "plk1": "PLK",
    "wee2": "WEE",
    "PKC_eta": "PKC",
    "CaMK1": "CAMK1",
    "CaMK1D": "CAMK1D",
    "CAMK1D iso2": "CAMK1D",
    "CaMK2": "CAMK2",
    "CaMK4": "CAMK4",
    "CaMK": "CAMK",
    "CaMKII": "CAMK2",
    "Chk1": "CHK1",
    "Chk2": "CHK2",
    "Csk": "CSK",
    "EPHA2": "EphA2",
    "EPHA4": "EphA4",
    "EPHB2": "EphB2",
    "Fgr": "FGR",
    "GSK-3": "GSK3",
    "Hck": "HCK",
    "MAPk14": "MAPK14",
    "Myt1": "MYT1",
    "p70S6Kb": "P70S6KB",
    "Pyk2": "PYK2",
}

kinases_to_ignore = {"BtrW", "C4", "C5", "C6", "CAD", "CARK1", "CERK1",
                     "CIPK11", "CIPK14", "CIPK21", "CIPK24", "CRPK1",
                     "EFR", "Etk", "BAK1", "FUS3", "G11", "Gin4", "GRIK1",
                     "GTF2F1", "GUCY2D", "HRR25", "HT1", "IPL1", "ISPK",
                     "KIN10", "KIN28", "LIT1", "MHCK", "MPS1", "PBL27", "PCK1",
                     "PDRP1", "PFKFB3", "PFKP", "PGK1", "PHO85", "PI3K",
                     "PIK3C2A", "PIK3CA", "PIK3CB", "PIK3CD", "PIK3CG",
                     "PIK3R1", "PIKFYVE", "zakA", "yopH", "viral", "unc-43", "tpa",
                     "sgg", "regulates", "pmk", "par", "max", "gck", "hop", "eIK1",
                     "eIK2", "PK4", "dsDNA", "cmkC", "ckk", "YpkA", "YAK1",
                     "VZV", "Tropomyosin kinase", "UL97", "Titin kinase", "Titin",
                     "TNNI3K", "SWE1", "STYK1", "STYK1", "STRK1", "STN7", "STK16", "STK19", "STK26", "STK33",
                     "SRC64B", "SRK2E", "SGV1", "RK", "PtkA", "PknA", "PknB", "PknD", "PknG", "PknH", "PknJ",
                     "PKH1", "PKH2", "ADCK5", "AFK", "ALDOA", "ALDOB", "ALPK1", "APP", "ASK7",
                     "ATPK1", "CDC15", "CDC28", "CDC5", "CIPK23", "CK10", "CKB", "CKM",
                     "CPK", "COL4A3BP", "CPK3", "GIN4", "HPrK/P", "wts", "CPK10"}

# Map the prefiltered kinases to their category in the kinbase phylogenetic tree

kinase_category_mapping = {
    "AAK1": "Other NAK AAK1",
    "Abl": "TK Abl ABL",
    "Ack": "TK Ack ACK",
    "AKT": "AGC PKB/AKT",
    "Akt1": "AGC AKT AKT1",
    "Akt2": "AGC AKT AKT2",
    "Akt3": "AGC AKT AKT3",
    "ALK": "TK Alk ALK",
    "ALK1": "TKL STKR Type1 ALK1",
    "ALK4": "TKL STKR Type1 ALK4",
    "AMPK": "CAMK CAMKL AMPK",
    "AMPKA1": "CAMK CAMKL AMPK AMPKa1",
    "AMPKA2": "CAMK CAMKL AMPK AMPKa2",
    "AMPKB1": "CAMK CAMKL AMPK",
    "AMPKG2": "CAMK CAMKL AMPK",
    "ANKRD3": "TKL RIPK ANKRD3",
    "ARAF": "TKL RAF ARAF",
    "Arg": "TK Abl ARG",
    "ASK1": "STE STE11 MAP3K5",  # http://kinase.com/web/current/kinbase/gene/5692
    "ATM": "Atypical PIKK ATM",  # http://kinase.com/web/current/kinbase/gene/5436
    "ATR": "Atypical PIKK ATR",  # http://kinase.com/web/current/kinbase/gene/5437
    "AurA": "Other AUR AurA",
    "AurB": "Other AUR AurB",
    "AurC": "Other AUR AurC",
    "AXL": "TK Axl AXL",
    "BCKDK": "Atypical PDHK BCKDK",  # http://kinase.com/web/current/kinbase/gene/5447
    "BIKE": "Other NAK BIKE",
    "BLK": "TK Src BLK",
    "BMPR1B": "TKL STKR Type1 BMPR1B",
    "BMX": "TK Tec BMX",
    "BRAF": "TKL RAF BRAF",
    "BRD4": "Atypical BRD BRD4",
    "BRK": "TK Src BRK",
    "BRSK1": "CAMK CAMKL BRSK BRSK1",
    "BRSK2": "CAMK CAMKL BRSK BRSK2",
    "BTK": "TK Tec BTK",
    "BUB1": "Other BUB BUB1	",
    "BVR": "NA",  # 3
    "CAK": "TK DDR DDR1",  # http://kinase.com/web/current/kinbase/gene/5548
    "CAMK": "CAMK",
    "CAMK1": "CAMK CAMK1 CAMK1A",
    "CAMK1D": "CAMK CAMK1 CAMK1D",
    "CAMK2": "CAMK CAMK2",
    "CaMK2A": "CAMK CAMK2 CaMK2a",
    "CaMK2D": "CAMK CAMK2 CaMK2d",
    "CaMK2G": "CAMK CAMK2 CaMK2g",
    "CAMK4": "CAMK CAMK4",
    "CaMKK1": "Other CAMKK Meta CaMKK1",
    "CaMKK2": "Other CAMKK Meta CaMKK2",
    "CASK": "CAMK CASK CASK",
    "CCRK": "CMGC CDK CCRK",
    "CDC2": "CMGC CDK CDK1/CDC2",
    # https://www.uniprot.org/uniprotkb/Q9Y5S2/entry  http://kinase.com/web/current/kinbase/gene/5752
    "CDC42BP": "AGC DMPK GEK MRCKb",
    "CDC7": "Other CDC7 CDC7",
    "CDK": "CMGC CDK",
    "CDK1": "CMGC CDK CDK1/CDC2",
    "CDK10": "CMGC CDK CDK10",
    "CDK11": "CMGC CDK CDK11",
    "CDK12": "CMGC CDK CDK12",
    "CDK13": "CMGC CDK CDK13",
    "CDK14": "CMGC CDK CDK14",
    "CDK15": "CMGC CDK CDK15",
    "CDK16": "CMGC CDK CDK16",
    "CDK18": "CMGC CDK CDK18",
    "CDK19": "CMGC CDK CDK19",
    "CDK2": "CMGC CDK CDK2",
    "CDK20": "CMGC CDK CDK20",
    "CDK3": "CMGC CDK CDK3",
    "CDK4": "CMGC CDK CDK4",
    "CDK5": "CMGC CDK CDK5",
    "CDK6": "CMGC CDK CDK6",
    "CDK7": "CMGC CDK CDK7",
    "CDK8": "CMGC CDK CDK8",
    "CDK9": "CMGC CDK CDK9",
    "CDKA": "CMGC CDK CDKA",
    # http://kinase.com/web/current/kinbase/gene/5506
    "Chak1": "Atypical Alpha ChaK CHAK1",
    "CHK": "CAMK",
    "CHK1": "CAMK CAMKL CHK1 CHK1",
    "CHK2": "CAMK RAD53 CHK2",
    "CK": "NA",  # Ambiguous CK1, CK2 in different groups
    "CK1": "CK1 CK1",
    "CK2": "Other CK2",
    "CK2A1": "Other CK2 CK2a1",
    "CLK1": "CMGC CLK CLK1",
    "CLK2": "CMGC CLK CLK2",
    "CLK3": "CMGC CLK CLK3",
    "CLK4": "CMGC CLK CLK4",
    "CRIK": "AGC DMPK CRIK",
    "CSK": "TK Csk CSK",
    "CTK": "TK Csk CTK",
    "DAPK": "CAMK DAPK",
    "DAPK1": "CAMK DAPK DAPK1",
    "DAPK2": "CAMK DAPK DAPK2",
    "DAPK3": "CAMK DAPK DAPK3",
    "DCAF1": "NA",
    "DCAMKL1": "CAMK DCAMKL DCAMKL1",
    "DDR1": "TK DDR DDR1",
    "DLK": "TKL MLK LZK DLK",
    "DMPK": "AGC DMPK GEK",
    "DMPK1": "AGC DMPK GEK DMPK1",
    "DNAPK": "Atypical PIKK DNAPK",  # http://kinase.com/web/current/kinbase/gene/5553
    "DRAK1": "CAMK DAPK DRAK1",
    "DRAK2": "CAMK DAPK DRAK2",
    "DYRK": "CMGC DYRK",
    "DYRK1A": "CMGC DYRK Dyrk1 DYRK1A",
    "DYRK1B": "CMGC DYRK Dyrk1 DYRK1B",
    "DYRK2": "CMGC DYRK Dyrk2 DYRK2",
    "DYRK3": "CMGC DYRK Dyrk2 DYRK3",
    "DYRK4": "CMGC DYRK Dyrk2 DYRK4",
    "EGFR": "TK EGFR EGFR",
    "ENPP1": "NA",  # 1
    "ENPP3": "NA",  # 1
    "EphA1": "TK Eph EphA1",
    "EphA2": "TK Eph EphA2",
    "EphA3": "TK Eph EphA3",
    "EphA4": "TK Eph EphA4",
    "EphA5": "TK Eph EphA5",
    "EphA6": "TK Eph EphA6",
    "EphA7": "TK Eph EphA7",
    "EphA8": "TK Eph EphA8",
    "EphB1": "TK Eph EphB1",
    "EphB2": "TK Eph EphB2",
    "EphB3": "TK Eph EphB3",
    "EphB5": "TK Eph",
    "EphB6": "TK Eph EphB6",
    "ErbB2": "TK EGFR HER2/ErbB2",
    "ERK1": "CMGC MAPK ERK Erk1",
    "ERK2": "CMGC MAPK ERK Erk2",
    "ERK3": "CMGC MAPK ERK Erk3",
    "ERK4": "CMGC MAPK ERK Erk4",
    "ERK5": "CMGC MAPK ERK Erk5",
    "ERK7": "CMGC MAPK Erk7 Erk7",
    "FAK": "TK Fak FAK",
    "FAM20C": "PKL FJ FAM FAM20C",  # http://kinase.com/web/current/kinbase/gene/9480
    "FER": "TK Fer FER",
    "FES": "TK Fer FES",
    "FGFR": "TK FGFR",
    "FGFR1": "TK FGFR FGFR1",
    "FGFR2": "TK FGFR FGFR2",
    "FGFR3": "TK FGFR FGFR3",
    "FGFR4": "TK FGFR FGFR4",
    "FGR": "TK Src FGR",
    "FLT1": "TK VEGFR FLT1",
    "FLT3": "TK PDGFR FLT3",
    "FLT4": "TK VEGFR FLT4",
    "FMS": "TK PDGFR FMS",
    "FRK": "TK Src FRK",
    "fu": "Other ULK Fused",
    "Fyn": "TK Src FYN",
    "GAK": "Other NAK GAK",
    "GCK": "STE STE20 KHS GCK",
    # There are 2 with same name??? Current classification sais this one http://kinase.com/web/current/kinbase/gene/5612
    "GCN2": "Other PEK GCN2 GCN2",
    "GPRK7": "AGC GRK GRK GPRK7",
    "GRK": "AGC GRK GRK",
    "GRK1": "AGC GRK GRK1",
    "GRK2": "AGC GRK GRK2",
    "GRK3": "AGC GRK GRK3",
    "GRK4": "AGC GRK GRK4",
    "GRK5": "AGC GRK GRK5",
    "GRK6": "AGC GRK GRK6",
    "GRK7": "AGC GRK GRK7",
    "GRP78": "NA",  # 1
    "GSK3": "CMGC GSK",
    "GSK3A": "CMGC GSK GSK3A",
    "GSK3B": "CMGC GSK GSK3B",
    "HASPIN": "Other Haspin Haspin",
    "HCK": "TK Src HCK",
    "HER2": "TK EGFR HER2/ErbB2",
    "HER4": "TK EGFR HER4/ErbB4",
    "HGK": "STE STE20 MSN ZC1/HGK",
    "HIPK1": "CMGC DYRK HIPK HIPK1",
    "HIPK2": "CMGC DYRK HIPK HIPK2",
    "HIPK3": "CMGC DYRK HIPK HIPK3",
    "HIPK4": "CMGC DYRK HIPK HIPK4",
    "HPK1": "STE STE20 KHS HPK1",
    "HRI": "Other PEK HRI",
    "HUNK": "CAMK CAMKL HUNK HUNK",
    "ICK": "CMGC RCK ICK",
    "IGF1R": "TK InsR IGF1R",
    "IKK": "Other IKK",
    "IKKA": "Other IKK IKKa",
    "IKKB": "Other IKK IKKb",
    "IKKE": "Other IKK IKKe",
    "ILK": "TKL MLK ILK ILK",
    "INSR": "TK InsR INSR",
    "IRAK1": "TKL IRAK IRAK1",
    "IRAK4": "TKL IRAK IRAK4",
    "IRE1": "Other IRE IRE1",
    "ITK": "TK Tec ITK",
    "JAK": "TK Jak",   # JAKs have multiple matches. current classification: http://kinase.com/web/current/kinbase/gene/5650
    "JAK1": "TK Jak JAK1",
    "JAK2": "TK Jak JAK2",
    "JAK3": "TK Jak JAK3",
    "JMJD6": "NA",  # 1
    "JNK": "CMGC MAPK JNK",
    "JNK1": "CMGC MAPK JNK JNK1",
    "JNK2": "CMGC MAPK JNK JNK2",
    "JNK3": "CMGC MAPK JNK JNK3",
    "KDR": "TK VEGFR KDR",
    "KHS1": "STE STE20 KHS KHS1",
    "KHS2": "STE STE20 KHS KHS2",
    "KIS": "Other Other-Unique KIS",
    "Kit": "TK PDGFR KIT",
    "KSR": "TKL RAF",
    "KSR1": "TKL RAF KSR1",
    "KSR2": "TKL RAF KSR2",
    "LATS1": "AGC NDR LATS1",
    "LATS2": "AGC NDR LATS2",
    "LCK": "TK Src LCK",
    "LIMK1": "TKL LISK LIMK LIMK1",
    "LIMK2": "TKL LISK LIMK LIMK2",
    # http://kinase.com/web/current/kinbase/gene/4460  (it's from nematode worm)
    "lit-1": "CMGC MAPK",
    "LKB1": "CAMK CAMKL LKB LKB1",
    "Lmr2": "TK Lmr LMR2",
    "LOK": "STE STE20 SLK LOK",
    "LRRK1": "TKL LRRK LRRK1",
    "LRRK2": "TKL LRRK LRRK2",
    "LTK": "TK Alk LTK",
    "LYN": "TK Src LYN",
    "LZK": "TKL MLK LZK LZK",
    "MAK": "CMGC RCK MAK",
    "MAP2K": "STE STE7",
    "MAP2K1": "STE STE7 MAP2K1",
    "MAP2K2": "STE STE7 MAP2K2",
    "MAP2K3": "STE STE7 MAP2K3",
    "MAP2K4": "STE STE7 MAP2K4",
    "MAP2K5": "STE STE7 MAP2K5",
    "MAP2K6": "STE STE7 MAP2K6",
    "MAP2K7": "STE STE7 MAP2K7",
    "MAP3K": "STE STE11",
    "MAP3K1": "STE STE11 MAP3K1	",
    "MAP3K10": "STE STE11",
    "MAP3K11": "STE STE11",
    "MAP3K14": "STE STE11",
    "MAP3K2": "STE STE11 MAP3K2",
    "MAP3K20": "STE STE11",
    "MAP3K3": "STE STE11 MAP3K3",
    "MAP3K4": "STE STE11 MAP3K4",
    "MAP3K5": "STE STE11 MAP3K5",
    "MAP3K6": "STE STE11 MAP3K6",
    "MAP3K7": "STE STE11 MAP3K7",
    "MAP3K8": "STE STE11 MAP3K8",
    "MAP4K1": "STE",
    "MAP4K2": "STE",
    "MAP4K4": "STE",
    "MAPK": "CMGC MAPK",
    "MAPK1": "CMGC MAPK p38",
    "MAPK10": "CMGC MAPK JNK JNK3",
    "MAPK11": "CMGC MAPK p38 p38b",
    "MAPK12": "CMGC MAPK ERK",
    "MAPK13": "CMGC MAPK ERK",
    "MAPK14": "CMGC MAPK p38 p38a/MAPK14",
    "MAPK3": "CMGC MAPK ERK Erk1",
    "MAPK4": "CMGC MAPK ERK Erk4",
    "MAPK6": "CMGC MAPK ERK Erk3",
    "MAPK7": "CMGC MAPK ERK Erk5",
    "MAPK8": "CMGC MAPK JNK JNK1",
    "MAPK9": "CMGC MAPK JNK JNK2",
    "MAPKAPK2": "CAMK MAPKAPK MAPKAPK MAPKAPK2",
    "MAPKAPK3": "CAMK MAPKAPK MAPKAPK MAPKAPK3",
    "MAPKAPK5": "CAMK MAPKAPK MAPKAPK MAPKAPK5",
    "MARK": "CAMK CAMKL MARK",
    "MARK1": "CAMK CAMKL MARK MARK1",
    "MARK2": "CAMK CAMKL MARK MARK2",
    "MARK3": "CAMK CAMKL MARK MARK3",
    "MARK4": "CAMK CAMKL MARK MARK4",
    "MAST3": "AGC MAST MAST3",
    "MAST4": "AGC MAST MAST4",
    "MASTL": "AGC MAST MASTL",
    "MELK": "CAMK CAMKL MELK MELK",
    "Mer": "TK Axl MER",
    "MET": "TK Met MET",
    "MINK": "STE STE20 MSN ZC3/MINK",
    "MINK1": "STE STE20 MSN ZC3/MINK",
    "MLCK": "CAMK MLCK",
    "MLK2": "TKL MLK MLK MLK2",
    "MLK3": "TKL MLK MLK MLK3",
    "MLK4": "TKL MLK MLK MLK4",
    "Mnk": "CAMK MAPKAPK MNK",
    "Mnk1": "CAMK MAPKAPK MNK MNK1",
    "Mnk2": "CAMK MAPKAPK MNK MNK2",
    "MOS": "Other MOS MOS",
    "MPSK1": "Other NAK MPSK1",
    "MRCKA": "AGC DMPK GEK MRCKa",
    "MRCKB": "AGC DMPK GEK MRCKb",
    # Multiple match. Selected by current classification. http://kinase.com/web/current/kinbase/gene/5755
    "MSK1": "AGC RSK MSK MSK1",
    "MSK2": "AGC RSK MSK MSK2",
    "MST1": "STE STE20 MST MST1",
    "MST2": "STE STE20 MST MST2",
    "MST3": "STE STE20 YSK MST3",
    "MST4": "STE STE20 YSK MST4",
    "mTOR": "Atypical PIKK FRAP mTOR",
    "MUSK": "TK Musk MUSK",
    "MYO3A": "STE STE20 NinaC MYO3A",
    "MYO3B": "STE STE20 NinaC MYO3B",
    "MYT1": "Other WEE MYT1",
    "NDR1": "AGC NDR NDR1",
    "NDR2": "AGC NDR NDR2",
    "NEK1": "Other NEK NEK1",
    "NEK10": "Other NEK NEK10",
    "NEK11": "Other NEK NEK11",
    "NEK2": "Other NEK NEK2",
    "NEK3": "Other NEK NEK3",
    "NEK5": "Other NEK NEK5",
    "NEK6": "Other NEK NEK6",
    "NEK7": "Other NEK NEK7",
    "NEK9": "Other NEK NEK9",
    "Nik": "STE STE-Unique NIK",
    "NLK": "CMGC MAPK nmo NLK",
    "NME1": "Atypical NDK NME NME1",
    "NME2": "Atypical NDK NME NME2",
    "NRK": "STE STE20 MSN ZC4/NRK",
    "NUAK1": "CAMK CAMKL NuaK NuaK1",
    "NUAK2": "CAMK CAMKL NuaK NuaK2",
    "OSR1": "STE STE20 FRAY OSR1",
    "CLA4": "STE STE20 PAKA CLA4",
    "p38": "CMGC MAPK p38",
    "P38A": "CMGC MAPK p38 p38a/MAPK14",
    "P38B": "CMGC MAPK p38 p38b",
    "P38D": "CMGC MAPK p38 p38d",
    "P38G": "CMGC MAPK p38 p38g",
    "CDKL2": "CMGC CDKL CDKL2",
    "p70S6K": "AGC RSK p70 p70S6K",
    "P70S6KB": "AGC RSK p70 p70S6Kb",
    "PAK1": "STE STE20 PAKA PAK1",
    "PAK2": "STE STE20 PAKA PAK2",
    "PAK3": "STE STE20 PAKA PAK3",
    "PAK4": "STE STE20 PAKB PAK4",
    "PAK5": "STE STE20 PAKB PAK5",
    "PAK6": "STE STE20 PAKB PAK6",
    "PAKA": "STE STE20 PAKA",
    "PASK": "CAMK CAMKL PASK PASK",
    "PBK": "Other TOPK PBK",
    "PCTAIRE1": "CMGC CDK TAIRE PCTAIRE1",
    "PDGFR": "TK PDGFR",
    "PDGFRA": "TK PDGFR PDGFRa",
    "PDGFRB": "TK PDGFR PDGFRb",
    "PDK1": "AGC PKB/AKT PDK1",
    "PDK2": "AGC PKB/AKT",
    "PDK3": "AGC PKB/AKT",
    "PDK4": "AGC PKB/AKT",
    "PDKC": "AGC PKC Delta PKCd",
    "PHK": "CAMK PHK",
    "PIM1": "CAMK PIM PIM1",
    "PIM2": "CAMK PIM",
    "PIM3": "CAMK PIM",
    "PINK1": "Other NKF2 PINK1",
    "PKA": "AGC PKA",
    "Pka-C1": "AGC PKA",
    "PKACa": "AGC PKA PKACa",
    "PKACb": "AGC PKA PKACb",
    "PKB": "AGC PKB/AKT",
    "PKC": "AGC PKC",
    "PKCa": "AGC PKC Alpha PKCa",
    "PKCb": "AGC PKC Alpha PKCb",
    "PKCd": "AGC PKC Delta PKCd",
    "PKCe": "AGC PKC Eta PKCe",
    "PKCg": "AGC PKC Alpha PKCg",
    "PKCh": "AGC PKC Eta PKCh",
    "PKCi": "AGC PKC Iota PKCi",
    "PKCt": "AGC PKC Delta PKCt",
    "PKCz": "AGC PKC Iota PKCz",
    "PKD": "CAMK PKD",
    "PKD1": "CAMK PKD PKD1",
    "PKD2": "CAMK PKD PKD2",
    "PKD3": "CAMK PKD PKD3",
    "PKG": "AGC PKG",
    "PKG1": "AGC PKG PKG1",
    "PKG2": "AGC PKG PKG2",
    "PKM": "NA",
    "PKN1": "AGC PKN PKN1",
    "PKN2": "AGC PKN PKN2",
    "PKN3": "AGC PKN PKN3",
    "PKR": "Other PEK PKR",
    "PLK": "Other PLK",
    "PLK1": "Other PLK PLK1",
    "PLK2": "Other PLK PLK2",
    "PLK3": "Other PLK PLK3",
    "PLK4": "Other PLK PLK4",
    "PRK1": "CAMK",
    "PRKC2": "CAMK PKC",
    "PRKDC": "Other VPS15 PIK3R4",
    "PRKX": "AGC PKA PRKX",
    "PRP4": "CMGC DYRK PRP4 PRP4",
    "PRPK": "Other Bud32 PRPK",
    "PYK2": "TK Fak PYK2",
    "QIK": "CAMK CAMKL QIK QIK",
    "QSK": "CAMK CAMKL QIK QSK",
    "RAF": "TKL RAF",
    "RAF1": "TKL RAF RAF1",
    "RET": "TK Ret RET",
    "RIOK1": "Atypical RIO RIOK1",
    "RIOK3": "Atypical RIO RIOK3",
    "RIPK1": "TKL RIPK RIPK1",
    "RIPK2": "TKL RIPK RIPK2",
    "RIPK3": "TKL RIPK RIPK3",
    "ROCK": "AGC DMPK ROCK",
    "ROCK1": "AGC DMPK ROCK ROCK1",
    "ROCK2": "AGC DMPK ROCK ROCK2",
    "RON": "TK Met RON",
    "ROR1": "TK Ror ROR1",
    "RSK": "AGC RSK RSK",
    "RSK-3": "AGC RSK RSK RSK3",
    "RSK1": "AGC RSK RSK RSK1",
    "RSK2": "AGC RSK RSK RSK2",
    "RSK3": "AGC RSK RSK RSK3",
    "RSK4": "AGC RSK RSK RSK4",
    "RSK5": "AGC RSK RSK RSK5",
    "SBK": "Other NKF1 SBK",
    "SGK": "AGC SGK",
    "SGK2": "AGC SGK SGK2",
    "SGK3": "AGC SGK SGK3",
    "SgK493": "Other Other-Unique SgK493",
    "SIK": "CAMK CAMKL QIK SIK",
    "SLK": "STE STE20 SLK SLK",
    "SMG1": "Atypical PIKK SMG1",
    "SPEG": "CAMK Trio SPEG",
    "SRC": "TK Src",
    "SRM": "TK Src SRM",
    "SRPK1": "CMGC SRPK SRPK1",
    "SRPK2": "CMGC SRPK SRPK2",
    "STK9": "CMGC CDKL CDKL5",  # http://kinase.com/web/current/kinbase/gene/5504
    "STLK3": "STE STE20 FRAY STLK3",
    "SYK": "TK Syk SYK",
    "TAF1": "Atypical TAF1",
    "TAK1": "TKL MLK TAK1 TAK1",
    "TAO": "STE STE20 TAO",
    "TAO1": "STE STE20 TAO TAO1",
    "TAO2": "STE STE20 TAO TAO2",
    "TAO3": "STE STE20 TAO TAO3",
    "TBK1": "Other IKK TBK1",
    "TEC": "TK Tec TEC",
    "TESK1": "TKL LISK TESK TESK1",
    "TESK2": "TKL LISK TESK TESK2",
    "TGFbR1": "TKL STKR Type1 TGFbR1",
    "TGFbR2": "TKL STKR Type2 TGFbR2",
    "TGM2": "NA",
    "TIE1": "TK Tie TIE1",
    "TIE2": "TK Tie TIE2",
    "TK": "TK",
    "TLK1": "Other TLK TLK1",
    "TLK2": "Other TLK TLK2",
    "TNIK": "STE STE20 MSN ZC2/TNIK",
    "TNK2": "TK Ack ACK",  # http://kinase.com/web/current/kinbase/gene/5409
    "TOR": "Atypical PIKK FRAP TOR",
    "Trb2": "CAMK Trbl Trb2",
    "TRKA": "TK Trk TRKA",
    "TRKB": "TK Trk TRKB",
    "TRKC": "TK Trk TRKC",
    # http://kinase.com/web/current/kinbase/gene/5506
    "TRPM7": "Atypical Alpha ChaK CHAK1",
    "TSSK1": "CAMK TSSK TSSK1",
    "TSSK2": "CAMK TSSK TSSK2",
    "TSSK3": "CAMK TSSK TSSK3",
    "TSSK4": "CAMK TSSK TSSK4",
    "TTBK1": "CK1 TTBK TTBK1",
    "TTBK2": "CK1 TTBK TTBK2",
    "TTK": "Other TTK TTK",
    "TXK": "TK Tec TXK",
    "TYK2": "TK JakA TYK2",
    "Tyro3": "TK Axl TYRO3",
    "ULK1": "Other ULK ULK1",
    "ULK2": "Other ULK ULK2",
    "ULK3": "Other ULK ULK3",
    "VEGFR": "TK VEGFR",
    "VPRBP": "NA",  # 1
    "VRK1": "CK1 VRK VRK1",
    "VRK2": "CK1 VRK VRK2",
    "VRK3": "CK1 VRK VRK3",
    "WEE": "Other WEE",
    "Wee1": "Other WEE Wee1",
    "Wee1b": "Other WEE Wee1B",
    "WEE2": "Other WEE",  # https://www.uniprot.org/uniprotkb/P0C1S8/entry
    "WNK1": "Other Wnk Wnk1",
    "WNK2": "Other Wnk Wnk2",
    "WNK3": "Other Wnk Wnk3",
    "WNK4": "Other Wnk Wnk4",
    "WSTF": "NA",  # 6
    "YES": "TK Src YES",
    "YSK1": "STE STE20 YSK YSK1",
    "ZAK": "TKL MLK MLK ZAK",
    "ZAP70": "TK Syk ZAP70",
}
