/* This file generated automatically. */
/*          Do not modify.            */
#include "udf.h"
#include "prop.h"
#include "dpm.h"
extern DEFINE_EXECUTE_ON_LOADING(mc_full_requirements, libname);
extern DEFINE_SPECIFIC_HEAT(mc_specific_heat, T, Tref, h, yi);
extern DEFINE_PROPERTY(mc_density, c, t);
extern DEFINE_PROPERTY(mc_speed_of_sound, c, t);
extern DEFINE_ADJUST(mc_manifold_project, domain);
extern DEFINE_ON_DEMAND(mc_decode_fields_on_demand);
extern DEFINE_SOURCE(mc_energy_source, c, t, dS, eqn);
extern DEFINE_PROPERTY(mc_viscosity, c, t);
extern DEFINE_PROPERTY(mc_thermal_conductivity, c, t);
extern DEFINE_SOURCE(mc_latent_uds_0_source, c, t, dS, eqn);
extern DEFINE_SOURCE(mc_latent_uds_1_source, c, t, dS, eqn);
extern DEFINE_SOURCE(mc_latent_uds_2_source, c, t, dS, eqn);
extern DEFINE_SOURCE(mc_latent_uds_3_source, c, t, dS, eqn);
extern DEFINE_SOURCE(mc_latent_uds_4_source, c, t, dS, eqn);
extern DEFINE_SOURCE(mc_latent_uds_5_source, c, t, dS, eqn);
extern DEFINE_SOURCE(mc_latent_uds_6_source, c, t, dS, eqn);
extern DEFINE_SOURCE(mc_latent_uds_7_source, c, t, dS, eqn);
__declspec(dllexport) UDF_Data udf_data[] = {
{"mc_full_requirements", (void(*)())mc_full_requirements, UDF_TYPE_EXECUTE_ON_LOADING},
{"mc_specific_heat", (void(*)())mc_specific_heat, UDF_TYPE_SPECIFIC_HEAT},
{"mc_density", (void(*)())mc_density, UDF_TYPE_PROPERTY},
{"mc_speed_of_sound", (void(*)())mc_speed_of_sound, UDF_TYPE_PROPERTY},
{"mc_manifold_project", (void(*)())mc_manifold_project, UDF_TYPE_ADJUST},
{"mc_decode_fields_on_demand", (void(*)())mc_decode_fields_on_demand, UDF_TYPE_ON_DEMAND},
{"mc_energy_source", (void(*)())mc_energy_source, UDF_TYPE_SOURCE},
{"mc_viscosity", (void(*)())mc_viscosity, UDF_TYPE_PROPERTY},
{"mc_thermal_conductivity", (void(*)())mc_thermal_conductivity, UDF_TYPE_PROPERTY},
{"mc_latent_uds_0_source", (void(*)())mc_latent_uds_0_source, UDF_TYPE_SOURCE},
{"mc_latent_uds_1_source", (void(*)())mc_latent_uds_1_source, UDF_TYPE_SOURCE},
{"mc_latent_uds_2_source", (void(*)())mc_latent_uds_2_source, UDF_TYPE_SOURCE},
{"mc_latent_uds_3_source", (void(*)())mc_latent_uds_3_source, UDF_TYPE_SOURCE},
{"mc_latent_uds_4_source", (void(*)())mc_latent_uds_4_source, UDF_TYPE_SOURCE},
{"mc_latent_uds_5_source", (void(*)())mc_latent_uds_5_source, UDF_TYPE_SOURCE},
{"mc_latent_uds_6_source", (void(*)())mc_latent_uds_6_source, UDF_TYPE_SOURCE},
{"mc_latent_uds_7_source", (void(*)())mc_latent_uds_7_source, UDF_TYPE_SOURCE},
};
__declspec(dllexport) int n_udf_data = sizeof(udf_data)/sizeof(UDF_Data);
#include "version.h"
__declspec(dllexport) void UDF_Inquire_Release(int *major, int *minor, int *revision)
{
  *major = RampantReleaseMajor;
  *minor = RampantReleaseMinor;
  *revision = RampantReleaseRevision;
}
