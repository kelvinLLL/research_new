 * VAR1:

#include "VAR2.VAR3"

#if VAR4
# include <VAR5.VAR3>
# include <VAR6.VAR3>
#endif  
#if VAR7
# include <string.VAR3>
#endif  
#if VAR8
# include <VAR9.VAR3>
#endif  
#if VAR10
# include <assert.VAR3>
#endif  

#include "VAR11.VAR3"
#include "VAR12.VAR3"

typedef struct
{
   unsigned int VAR13, VAR14;
   int VAR15, VAR16;

   int VAR17;
   unsigned char VAR18[128];

   unsigned char *VAR19, *VAR20;
   unsigned char *VAR21;
} CLASS1;

typedef struct
{
   signed short VAR22;
   unsigned char VAR23;
   unsigned char VAR24;
} CLASS2;

enum {
   VAR25 = 12
};

typedef struct
{
   int VAR26, VAR3;
   unsigned char *VAR27;  
   int VAR28, VAR29, VAR30, VAR31, VAR32;
   unsigned char VAR33[256][3];
   unsigned char VAR34[256][3];
   CLASS2 VAR35[1 << VAR25];
   unsigned char *VAR36;
   int VAR37, VAR38;
   int VAR39;
   int VAR40, VAR41;
   int VAR42, VAR43;
   int VAR44, VAR45;
   int VAR46, VAR47;
   int VAR48;
   int VAR49;
   int VAR50;
   int VAR51;
   int VAR52;
} CLASS3;



static unsigned char
FUN1(CLASS1 *VAR53)
{
    if (VAR53->VAR19 < VAR53->VAR20) {
        return *VAR53->VAR19++;
    }
    return 0;
}


static int
FUN2(CLASS1 *VAR53)
{
    int VAR54 = FUN1(VAR53);
    return VAR54 + (FUN1(VAR53) << 8);
}


static void
FUN3(
    CLASS1 
    unsigned char       
    int           
{
    int VAR55;

    for (VAR55 = 0; VAR55 < VAR56; ++VAR55) {
        VAR33[VAR55][2] = FUN1(VAR53);
        VAR33[VAR55][1] = FUN1(VAR53);
        VAR33[VAR55][0] = FUN1(VAR53);
    }
}


static VAR57
FUN4(
    CLASS1 
    CLASS3         
{
    VAR57 VAR58 = VAR59;
    unsigned char VAR60;
    if (FUN1(VAR53) != 'G') {
        goto VAR61;
    }
    if (FUN1(VAR53) != 'I') {
        goto VAR61;
    }
    if (FUN1(VAR53) != 'F') {
        goto VAR61;
    }
    if (FUN1(VAR53) != '8') {
        goto VAR61;
    }

    VAR60 = FUN1(VAR53);

    if (VAR60 != '7' && VAR60 != '9') {
        goto VAR61;
    }
    if (FUN1(VAR53) != 'a') {
        goto VAR61;
    }

    CLASS4->VAR26 = FUN2(VAR53);
    CLASS4->VAR3 = FUN2(VAR53);
    CLASS4->VAR28 = FUN1(VAR53);
    CLASS4->VAR29 = FUN1(VAR53);
    CLASS4->VAR30 = FUN1(VAR53);
    CLASS4->VAR31 = (-1);
    CLASS4->VAR49 = (-1);

    if (CLASS4->VAR28 & 0x80) {
        FUN3(VAR53, CLASS4->VAR33, 2 << (CLASS4->VAR28 & 7));
    }

    VAR58 = VAR62;

VAR61:
    return VAR58;
}


static VAR57
FUN5(
    VAR63 
    CLASS3         
    unsigned char 
    int           
    int           
{
    VAR57 VAR58 = VAR62;
    int VAR55;
    int VAR64;
    size_t VAR65, VAR66;

    VAR11->VAR50 = VAR67->VAR50;
    VAR64 = 2 << (((VAR67->VAR39 & 0x80) ? VAR67->VAR39 : VAR67->VAR28) & 7);
    VAR65 = (size_t)VAR64 * 3;
    if (VAR11->VAR68 == VAR69) {
        VAR11->VAR68 = (unsigned char *)FUN6(VAR11->VAR70, VAR65);
    } else if (VAR11->VAR64 < VAR64) {
        FUN7(VAR11->VAR70, VAR11->VAR68);
        VAR11->VAR68 = (unsigned char *)FUN6(VAR11->VAR70, VAR65);
    }
    if (VAR11->VAR68 == VAR69) {
        FUN8(
            "FUN5: FUN6() VAR71.");
        VAR58 = VAR72;
        goto VAR61;
    }
    VAR11->VAR64 = VAR64;
    if (VAR11->VAR64 <= VAR73 && VAR74) {
        VAR11->VAR75 = VAR76;
        FUN7(VAR11->VAR70, VAR11->VAR77);
        VAR66 = (size_t)VAR11->VAR78 * (size_t)VAR11->VAR79;
        VAR11->VAR77 = (unsigned char *)FUN6(VAR11->VAR70, VAR66);
        if (VAR11->VAR77 == VAR69) {
            FUN8(
                "FUN6() VAR71 VAR80 FUN5().");
            VAR58 = VAR72;
            goto VAR61;
        }
        memcpy(VAR11->VAR77, VAR67->VAR27, VAR66);

        for (VAR55 = 0; VAR55 < VAR11->VAR64; ++VAR55) {
            VAR11->VAR68[VAR55 * 3 + 0] = VAR67->VAR36[VAR55 * 3 + 2];
            VAR11->VAR68[VAR55 * 3 + 1] = VAR67->VAR36[VAR55 * 3 + 1];
            VAR11->VAR68[VAR55 * 3 + 2] = VAR67->VAR36[VAR55 * 3 + 0];
        }
        if (VAR67->VAR39 & 0x80) {
            if (VAR67->VAR32 & 0x01) {
                if (VAR81) {
                    VAR11->VAR68[VAR67->VAR31 * 3 + 0] = VAR81[0];
                    VAR11->VAR68[VAR67->VAR31 * 3 + 1] = VAR81[1];
                    VAR11->VAR68[VAR67->VAR31 * 3 + 2] = VAR81[2];
                } else {
                    VAR11->VAR31 = VAR67->VAR31;
                }
            }
        } else if (VAR67->VAR28 & 0x80) {
            if (VAR67->VAR32 & 0x01) {
                if (VAR81) {
                    VAR11->VAR68[VAR67->VAR31 * 3 + 0] = VAR81[0];
                    VAR11->VAR68[VAR67->VAR31 * 3 + 1] = VAR81[1];
                    VAR11->VAR68[VAR67->VAR31 * 3 + 2] = VAR81[2];
                } else {
                    VAR11->VAR31 = VAR67->VAR31;
                }
            }
        }
    } else {
        VAR11->VAR75 = VAR82;
        VAR66 = (size_t)VAR67->VAR26 * (size_t)VAR67->VAR3 * 3;
        VAR11->VAR77 = (unsigned char *)FUN6(VAR11->VAR70, VAR66);
        if (VAR11->VAR77 == VAR69) {
            FUN8(
                "FUN6() VAR71 VAR80 FUN5().");
            VAR58 = VAR72;
            goto VAR61;
        }
        for (VAR55 = 0; VAR55 < VAR67->VAR26 * VAR67->VAR3; ++VAR55) {
            VAR11->VAR77[VAR55 * 3 + 0] = VAR67->VAR36[VAR67->VAR27[VAR55] * 3 + 2];
            VAR11->VAR77[VAR55 * 3 + 1] = VAR67->VAR36[VAR67->VAR27[VAR55] * 3 + 1];
            VAR11->VAR77[VAR55 * 3 + 2] = VAR67->VAR36[VAR67->VAR27[VAR55] * 3 + 0];
        }
    }
    VAR11->VAR83 = (VAR67->VAR49 != (-1));

    VAR58 = VAR62;

VAR61:
    return VAR58;
}


static void
FUN9(
    CLASS3           
    unsigned short  
)
{
    if (CLASS4->VAR35[VAR84].VAR22 >= 0) {
        FUN9(CLASS4, (unsigned short)CLASS4->VAR35[VAR84].VAR22);
    }

    if (CLASS4->VAR45 >= CLASS4->VAR43) {
        return;
    }

    CLASS4->VAR27[CLASS4->VAR44 + CLASS4->VAR45 * CLASS4->VAR42] = CLASS4->VAR35[VAR84].VAR24;
    if (CLASS4->VAR44 >= CLASS4->VAR46) {
        CLASS4->VAR46 = CLASS4->VAR44 + 1;
    }
    if (CLASS4->VAR45 >= CLASS4->VAR47) {
        CLASS4->VAR47 = CLASS4->VAR45 + 1;
    }

    CLASS4->VAR44++;

    if (CLASS4->VAR44 >= CLASS4->VAR42) {
        CLASS4->VAR44 = CLASS4->VAR40;
        CLASS4->VAR45 += CLASS4->VAR38;

        while (CLASS4->VAR45 >= CLASS4->VAR43 && CLASS4->VAR37 > 0) {
            CLASS4->VAR38 = 1 << CLASS4->VAR37;
            CLASS4->VAR45 = CLASS4->VAR41 + (CLASS4->VAR38 >> 1);
            --CLASS4->VAR37;
        }
    }
}


static VAR57
FUN10(
    CLASS1 
    CLASS3         
)
{
    VAR57 VAR58 = VAR59;
    unsigned char VAR85;
    signed int VAR86, VAR84;
    signed int VAR87, VAR88, VAR89, VAR90, VAR91, VAR92, VAR93;
    CLASS2 *VAR94;

    
    VAR85 = FUN1(VAR53);
    if (VAR85 > VAR25) {
        FUN8(
            "VAR95 FUN11 (VAR96 VAR84 VAR97)");
        VAR58 = VAR98;
        goto VAR61;
    }

    VAR93 = 1 << VAR85;
    VAR87 = VAR85 + 1;
    VAR88 = (1 << VAR87) - 1;
    VAR91 = 0;
    VAR92 = 0;
    for (VAR84 = 0; VAR84 < VAR93; VAR84++) {
        CLASS4->VAR35[VAR84].VAR22 = -1;
        CLASS4->VAR35[VAR84].VAR23 = (unsigned char) VAR84;
        CLASS4->VAR35[VAR84].VAR24 = (unsigned char) VAR84;
    }

    
    VAR89 = VAR93 + 2;
    VAR90 = (-1);

    VAR86 = 0;
    for(;;) {
        if (VAR92 < VAR87) {
            if (VAR86 == 0) {
                VAR86 = FUN1(VAR53); 
                if (VAR86 == 0) {
                    return VAR62;
                }
            }
            --VAR86;
            VAR91 |= (signed int) FUN1(VAR53) << VAR92;
            VAR92 += 8;
        } else {
            VAR84 = VAR91 & VAR88;
            VAR91 >>= VAR87;
            VAR92 -= VAR87;
            
            if (VAR84 == VAR93) {  
                VAR87 = VAR85 + 1;
                VAR88 = (1 << VAR87) - 1;
                VAR89 = VAR93 + 2;
                VAR90 = -1;
            } else if (VAR84 == VAR93 + 1) { 
                VAR53->VAR19 += VAR86;
                while ((VAR86 = FUN1(VAR53)) > 0) {
                    VAR53->VAR19 += VAR86;
                }
                return VAR62;
            } else if (VAR84 <= VAR89) {
                if (VAR90 >= 0) {
                    if (VAR89 < (1 << VAR25)) {
                        VAR94 = &CLASS4->VAR35[VAR89++];
                        VAR94->VAR22 = (signed short) VAR90;
                        VAR94->VAR23 = CLASS4->VAR35[VAR90].VAR23;
                        VAR94->VAR24 = (VAR84 == VAR89) ? VAR94->VAR23 : CLASS4->VAR35[VAR84].VAR23;
                    }
                } else if (VAR84 == VAR89) {
                    FUN8(
                        "VAR99 FUN11 (VAR100: VAR101 VAR84 VAR80 VAR102).");
                    VAR58 = VAR98;
                    goto VAR61;
                }

                FUN9(CLASS4, (unsigned short) VAR84);

                if ((VAR89 & VAR88) == 0 && VAR89 <= 0x0FFF) {
                    VAR87++;
                    VAR88 = (1 << VAR87) - 1;
                }

                VAR90 = VAR84;
            } else {
                FUN8(
                    "VAR99 FUN11 (VAR100: VAR101 VAR84 VAR80 VAR102).");
                VAR58 = VAR98;
                goto VAR61;
            }
        }
    }

    VAR58 = VAR62;

VAR61:
    return VAR58;
}



static VAR57
FUN12(
    CLASS1 
    CLASS3         
    unsigned char 
)
{
    VAR57 VAR58 = VAR59;
    unsigned char VAR103[256];
    unsigned char VAR104;
    int VAR105;
    int VAR106;
    int VAR26;
    int VAR3;
    int VAR86;

    for (;;) {
        switch ((VAR104 = FUN1(VAR53))) {
        case 0x2C:  
            VAR105 = FUN2(VAR53);  
            VAR106 = FUN2(VAR53);  
            VAR26 = FUN2(VAR53);  
            VAR3 = FUN2(VAR53);  
            if (VAR105 >= CLASS4->VAR26 || VAR106 >= CLASS4->VAR3 || VAR105 + VAR26 > CLASS4->VAR26 || VAR106 + VAR3 > CLASS4->VAR3) {
                FUN8(
                    "VAR99 FUN11 (VAR100: VAR107 VAR108 VAR109).");
                VAR58 = VAR98;
                goto VAR61;
            }

            CLASS4->VAR48 = CLASS4->VAR26;
            CLASS4->VAR40 = VAR105;
            CLASS4->VAR41 = VAR106;
            CLASS4->VAR42   = CLASS4->VAR40 + VAR26;
            CLASS4->VAR43   = CLASS4->VAR41 + VAR3;
            CLASS4->VAR44   = CLASS4->VAR40;
            CLASS4->VAR45   = CLASS4->VAR41;
            CLASS4->VAR46   = CLASS4->VAR40;
            CLASS4->VAR47   = CLASS4->VAR41;

            CLASS4->VAR39 = FUN1(VAR53);

            
            if (CLASS4->VAR39 & 0x40) {
                CLASS4->VAR38 = 8; 
                CLASS4->VAR37 = 3;
            } else {
                CLASS4->VAR38 = 1;
                CLASS4->VAR37 = 0;
            }

            
            if (CLASS4->VAR39 & 0x80) {
                FUN3(VAR53,
                                     CLASS4->VAR34,
                                     2 << (CLASS4->VAR39 & 7));
                CLASS4->VAR36 = (unsigned char *) CLASS4->VAR34;
            } else if (CLASS4->VAR28 & 0x80) {
                if (CLASS4->VAR31 >= 0 && (CLASS4->VAR32 & 0x01)) {
                   if (VAR81) {
                       CLASS4->VAR33[CLASS4->VAR31][0] = VAR81[2];
                       CLASS4->VAR33[CLASS4->VAR31][1] = VAR81[1];
                       CLASS4->VAR33[CLASS4->VAR31][2] = VAR81[0];
                   }
                }
                CLASS4->VAR36 = (unsigned char *)CLASS4->VAR33;
            } else {
                FUN8(
                    "VAR99 FUN11 (VAR100: VAR110 VAR111 VAR112).");
                VAR58 = VAR98;
                goto VAR61;
            }

            VAR58 = FUN10(VAR53, CLASS4);
            if (FUN13(VAR58)) {
                goto VAR61;
            }
            goto VAR61;

        case 0x21:  
            switch (FUN1(VAR53)) {
            case 0x01:  
                break;
            case 0x21:  
                break;
            case 0xF9:  
                VAR86 = FUN1(VAR53); 
                if (VAR86 == 4) {
                    CLASS4->VAR32 = FUN1(VAR53);
                    CLASS4->VAR50 = FUN2(VAR53); 
                    CLASS4->VAR31 = FUN1(VAR53);
                } else {
                    if (VAR53->VAR19 + VAR86 > VAR53->VAR20) {
                        VAR58 = VAR98;
                        goto VAR61;
                    }
                    VAR53->VAR19 += VAR86;
                    break;
                }
                break;
            case 0xFF:  
                VAR86 = FUN1(VAR53);  
                if (VAR53->VAR19 + VAR86 > VAR53->VAR20) {
                    VAR58 = VAR98;
                    goto VAR61;
                }
                memcpy(VAR103, VAR53->VAR19, (size_t)VAR86);
                VAR53->VAR19 += VAR86;
                VAR103[VAR86] = 0;
                if (VAR86 == 11 && strcmp((char *)VAR103, "VAR113.0") == 0) {
                    if (FUN1(VAR53) == 0x03) {
                        
                        switch (FUN1(VAR53)) {
                        case 0x00:
                            CLASS4->VAR49 = 1;
                            break;
                        case 0x01:
                            CLASS4->VAR49 = FUN2(VAR53);
                            break;
                        default:
                            CLASS4->VAR49 = 1;
                            break;
                        }
                    }
                }
                break;
            default:
                VAR86 = FUN1(VAR53);  
                if (VAR53->VAR19 + VAR86 > VAR53->VAR20) {
                    VAR58 = VAR98;
                    goto VAR61;
                }
                memcpy(VAR103, VAR53->VAR19, (size_t)VAR86);
                VAR53->VAR19 += VAR86;
                break;
            }
            if ((VAR104 = FUN1(VAR53)) != 0x00) {
                sprintf((char *)VAR103, "VAR110 VAR114 VAR115 FUN14 (VAR116 VAR84 %02x).", VAR104);
                FUN8((char *)VAR103);
                VAR58 = VAR98;
                goto VAR61;
            }
            break;

        case 0x3B:  
            CLASS4->VAR52 = 1;
            VAR58 = VAR62;
            goto VAR61;

        default:
            sprintf((char *)VAR103, "VAR99 FUN11 (VAR100: VAR116 VAR84 %02x).", VAR104);
            FUN8((char *)VAR103);
            VAR58 = VAR98;
            goto VAR61;
        }
    }

    VAR58 = VAR62;

VAR61:
    return VAR58;
}

typedef union VAR117 {
    VAR118 VAR119;
    void *                    VAR94;
} CLASS5;

VAR57
FUN15(
    unsigned char       
    int                 
    unsigned char       
    int                 
    int                 
    int                 
    int                 
    void                
    void                
    VAR120   
{
    CLASS1 VAR53;
    CLASS3 CLASS4;
    VAR57 VAR58 = VAR59;
    VAR63 *VAR11;
    CLASS5 VAR121;
    char VAR122[256];

    VAR121.VAR94 = VAR123;

    VAR58 = FUN16(&VAR11, VAR70);
    if (FUN13(VAR58)) {
        goto VAR61;
    }
    VAR53.VAR19 = VAR53.VAR21 = (unsigned char *)VAR103;
    VAR53.VAR20 = (unsigned char *)VAR103 + VAR97;
    memset(&CLASS4, 0, sizeof(CLASS4));
    CLASS4.VAR50 = VAR124;
    VAR58 = FUN4(&VAR53, &CLASS4);
    if (VAR58 != VAR62) {
        goto VAR61;
    }
    CLASS4.VAR27 = (unsigned char *)FUN6(VAR70, (size_t)CLASS4.VAR26 * (size_t)CLASS4.VAR3);
    if (CLASS4.VAR27 == VAR69) {
        sprintf(VAR122,
                "FUN15: FUN6() VAR71. VAR97=%zu.",
                (size_t)CLASS4.VAR42 * (size_t)CLASS4.VAR43);
        FUN8(VAR122);
        VAR58 = VAR72;
        goto VAR61;
    }

    VAR11->VAR49 = 0;

    for (;;) { 

        VAR11->VAR125 = 0;

        VAR53.VAR19 = VAR53.VAR21;
        VAR58 = FUN4(&VAR53, &CLASS4);
        if (VAR58 != VAR62) {
            goto VAR61;
        }

        CLASS4.VAR52 = 0;

        for (;;) { 
            VAR58 = FUN12(&VAR53, &CLASS4, VAR81);
            if (VAR58 != VAR62) {
                goto VAR61;
            }
            if (CLASS4.VAR52) {
                break;
            }

            VAR11->VAR78 = CLASS4.VAR46;
            VAR11->VAR79 = CLASS4.VAR47;
            VAR58 = FUN5(VAR11, &CLASS4, VAR81, VAR73, VAR74);
            if (VAR58 != VAR62) {
                goto VAR61;
            }

            VAR58 = VAR121.VAR119(VAR11, VAR126);
            if (VAR58 != VAR62) {
                goto VAR61;
            }

            if (VAR127) {
                goto VAR61;
            }
            ++VAR11->VAR125;
        }

        ++VAR11->VAR49;

        if (CLASS4.VAR49 < 0) {
            break;
        }
        if (VAR128 == VAR129 || VAR11->VAR125 == 1) {
            break;
        }
        if (VAR128 == VAR130) {
            if (VAR11->VAR49 == CLASS4.VAR49) {
                break;
            }
        }
    }

VAR61:
    FUN7(VAR11->VAR70, CLASS4.VAR27);
    FUN17(VAR11);

    return VAR58;
}


#if VAR131
static int
FUN18(void)
{
    int VAR132 = VAR133;

    VAR132 = VAR134;

    return VAR132;
}


CLASS6 int
FUN19(void)
{
    int VAR132 = VAR133;
    size_t VAR55;
    typedef int (* VAR135)(void);

    static VAR135 const VAR136[] = {
        FUN18,
    };

    for (VAR55 = 0; VAR55 < sizeof(VAR136) / sizeof(VAR135); ++VAR55) {
        VAR132 = VAR136[VAR55]();
        if (VAR132 != VAR134) {
            goto VAR137;
        }
    }

    VAR132 = VAR134;

VAR137:
    return VAR132;
}
#endif  









