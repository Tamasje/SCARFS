/* This file generated automatically. */
/*          Do not modify.            */
#include "udf.h"
#include "prop.h"
#include "dpm.h"
extern DEFINE_EXECUTE_AT_END(cluster_and_chemistry);
extern DEFINE_EXECUTE_AT_EXIT(report_timings);
extern DEFINE_SOURCE(S_H2, c, t, dS, eqn);
extern DEFINE_SOURCE(S_CH4,c,t,dS,eqn);
extern DEFINE_SOURCE(S_C2H6,c,t,dS,eqn);
extern DEFINE_SOURCE(S_C3H8,c,t,dS,eqn);
extern DEFINE_SOURCE(S_NC4H10,c,t,dS,eqn);
extern DEFINE_SOURCE(S_BENZENE,c,t,dS,eqn);
extern DEFINE_SOURCE(S_TOLUENE,c,t,dS,eqn);
extern DEFINE_SOURCE(S_XYLENES,c,t,dS,eqn);
extern DEFINE_SOURCE(S_NAPHTHALENE,c,t,dS,eqn);
extern DEFINE_SOURCE(S_CHD14,c,t,dS,eqn);
extern DEFINE_SOURCE(S_C2H4,c,t,dS,eqn);
extern DEFINE_SOURCE(S_C3H6,c,t,dS,eqn);
extern DEFINE_SOURCE(S_C4H6_1_3,c,t,dS,eqn);
extern DEFINE_SOURCE(S_C4H8_1,c,t,dS,eqn);
extern DEFINE_SOURCE(S_C4H8_2,c,t,dS,eqn);
extern DEFINE_SOURCE(S_C5DIOL_1_4,c,t,dS,eqn);
extern DEFINE_SOURCE(S_C5H10_1,c,t,dS,eqn);
extern DEFINE_SOURCE(S_C5H10_2,c,t,dS,eqn);
extern DEFINE_SOURCE(S_CPD,c,t,dS,eqn);
extern DEFINE_SOURCE(S_MECPD,c,t,dS,eqn);
extern DEFINE_SOURCE(S_STYRENE,c,t,dS,eqn);
extern DEFINE_SOURCE(S_MEINDENE,c,t,dS,eqn);
extern DEFINE_SOURCE(S_NAROLC11,c,t,dS,eqn);
extern DEFINE_SOURCE(S_C4H6_1_2,c,t,dS,eqn);
extern DEFINE_SOURCE(S_FULVENE,c,t,dS,eqn);
extern DEFINE_SOURCE(S_C4H4,c,t,dS,eqn);
extern DEFINE_SOURCE(S_CH2CPENE,c,t,dS,eqn);
extern DEFINE_SOURCE(S_C2H2,c,t,dS,eqn);
extern DEFINE_SOURCE(S_C3H4_MA,c,t,dS,eqn);
extern DEFINE_SOURCE(S_C3H4_PD,c,t,dS,eqn);
extern DEFINE_SOURCE(S_ENER,c,t,dS,eqn);
__declspec(dllexport) UDF_Data udf_data[] = {
{"cluster_and_chemistry", (void(*)())cluster_and_chemistry, UDF_TYPE_EXECUTE_AT_END},
{"report_timings", (void(*)())report_timings, UDF_TYPE_EXECUTE_AT_EXIT},
{"S_H2", (void(*)())S_H2, UDF_TYPE_SOURCE},
{"S_CH4", (void(*)())S_CH4, UDF_TYPE_SOURCE},
{"S_C2H6", (void(*)())S_C2H6, UDF_TYPE_SOURCE},
{"S_C3H8", (void(*)())S_C3H8, UDF_TYPE_SOURCE},
{"S_NC4H10", (void(*)())S_NC4H10, UDF_TYPE_SOURCE},
{"S_BENZENE", (void(*)())S_BENZENE, UDF_TYPE_SOURCE},
{"S_TOLUENE", (void(*)())S_TOLUENE, UDF_TYPE_SOURCE},
{"S_XYLENES", (void(*)())S_XYLENES, UDF_TYPE_SOURCE},
{"S_NAPHTHALENE", (void(*)())S_NAPHTHALENE, UDF_TYPE_SOURCE},
{"S_CHD14", (void(*)())S_CHD14, UDF_TYPE_SOURCE},
{"S_C2H4", (void(*)())S_C2H4, UDF_TYPE_SOURCE},
{"S_C3H6", (void(*)())S_C3H6, UDF_TYPE_SOURCE},
{"S_C4H6_1_3", (void(*)())S_C4H6_1_3, UDF_TYPE_SOURCE},
{"S_C4H8_1", (void(*)())S_C4H8_1, UDF_TYPE_SOURCE},
{"S_C4H8_2", (void(*)())S_C4H8_2, UDF_TYPE_SOURCE},
{"S_C5DIOL_1_4", (void(*)())S_C5DIOL_1_4, UDF_TYPE_SOURCE},
{"S_C5H10_1", (void(*)())S_C5H10_1, UDF_TYPE_SOURCE},
{"S_C5H10_2", (void(*)())S_C5H10_2, UDF_TYPE_SOURCE},
{"S_CPD", (void(*)())S_CPD, UDF_TYPE_SOURCE},
{"S_MECPD", (void(*)())S_MECPD, UDF_TYPE_SOURCE},
{"S_STYRENE", (void(*)())S_STYRENE, UDF_TYPE_SOURCE},
{"S_MEINDENE", (void(*)())S_MEINDENE, UDF_TYPE_SOURCE},
{"S_NAROLC11", (void(*)())S_NAROLC11, UDF_TYPE_SOURCE},
{"S_C4H6_1_2", (void(*)())S_C4H6_1_2, UDF_TYPE_SOURCE},
{"S_FULVENE", (void(*)())S_FULVENE, UDF_TYPE_SOURCE},
{"S_C4H4", (void(*)())S_C4H4, UDF_TYPE_SOURCE},
{"S_CH2CPENE", (void(*)())S_CH2CPENE, UDF_TYPE_SOURCE},
{"S_C2H2", (void(*)())S_C2H2, UDF_TYPE_SOURCE},
{"S_C3H4_MA", (void(*)())S_C3H4_MA, UDF_TYPE_SOURCE},
{"S_C3H4_PD", (void(*)())S_C3H4_PD, UDF_TYPE_SOURCE},
{"S_ENER", (void(*)())S_ENER, UDF_TYPE_SOURCE},
};
__declspec(dllexport) int n_udf_data = sizeof(udf_data)/sizeof(UDF_Data);
#include "version.h"
__declspec(dllexport) void UDF_Inquire_Release(int *major, int *minor, int *revision)
{
  *major = RampantReleaseMajor;
  *minor = RampantReleaseMinor;
  *revision = RampantReleaseRevision;
}
