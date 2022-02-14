/***************************************************************************
 # Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/

#include "Utils/Math/MathConstants.slangh"
#include "Scene/Material/MaterialDefines.slangh"
#include "Scene/Lights/Lights.slang"

import Scene.Raytracing;
import Utils.Math.MathHelpers;
import Scene.Lights.LightData;
import Experimental.Scene.Material.MaterialShading;
import Experimental.Scene.Lights.LightHelpers;
import Scene.TextureSampler;
import Scene.Material.MaterialData;
import Scene.HitInfo;
import RenderPasses.Shared.PathTracer.RayFootprintData;
import Scene.Material.BRDF;

// early out settings
#define TRANSMIT_EPSILON 0.001
#define REFLECT_EPSILON 0.001
#define SHADOW_MIN 0.2
#define AMBIENT_LIGHT 0.3
#define MAX_BOUNCES 16
#define MAX_INTERNAL_REFLECTIONS 8
#define TRANSPARENT_DEPTH 8

cbuffer CB
{
    uint gFrameCount; // Frame count since scene was loaded.
}

// Outputs
RWTexture2D<float4> gOutputColor;
RWTexture2D<float4> gDebug;
// Default ray payload
struct RayPayload
{
    PackedHitInfo hit; // for triangles: instanceID, primitiveIndex and compressed barycentrics. for billboards: x = aabbOffset
    float depth; // rayTCurrent of intersection
};

// Payload for spherical billboards
struct BillboardPayload
{
    uint spheres[TRANSPARENT_DEPTH];
    float tcurrent[TRANSPARENT_DEPTH];
    float4 leftover;
    float t_leftover;
    RayFootprint footprint; // ray differentials for texture filtering
    uint pos;
    float rayTMax; // ray T to next triangle
    float listMax; //transparent object furthest away
    float listMin; //closest possible transparent object
};


#include "Helper.slang" // contains general helper function to retrieve billboard vertex data, billboard tangent plane and shading related data
#include "Shadow.slang" // contains the trace shadow ray function (should be replaced with shadow maps for real-time)
#include "RayDifferentials.slang" // contains helper functions to modify the ray differentials and filtered shading textures


/****** Triangle Hit Shaders ******/

[shader("miss")]
void miss(inout RayPayload rayData : SV_RayPayload) {}

[shader("anyhit")]
void triangleAnyHit(
    uniform HitShaderParams hitParams,
    inout RayPayload rayData : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
    // Alpha test for non-opaque geometry.
    VertexData v = getVertexData(hitParams, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(hitParams.getGlobalHitID());
    if (alphaTest(v, gScene.materials[materialID], gScene.materialResources[materialID], 0.f)) IgnoreHit();
}

[shader("closesthit")]
void triangleClosestHit(
    uniform HitShaderParams hitParams,
    inout RayPayload rayData : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
    // store hit information (shading is done in ray generation shader)
    rayData.depth = RayTCurrent();
    HitInfo hit;
    hit.type = InstanceType::TriangleMesh;
    hit.instanceID = hitParams.getGlobalHitID();
    hit.primitiveIndex = PrimitiveIndex();
    hit.barycentrics = attribs.barycentrics;
    rayData.hit = hit.encode();
}


/****** Billboard Hit Shaders ******/
struct BoxIntersectionAttributes
{
    int aabbOffset;
};

[shader("intersection")]
void boxIntersect()
{
    const uint aabbOffset = gScene.proceduralPrimitives[GeometryIndex()].AABBOffset;
    const AABB bbox = gScene.proceduralPrimitiveAABBs[aabbOffset];
    const float2 dim = bbox.extent().xy; // w = dim.x, h = dim.y
    const float3 C = bbox.center();

    const float r = 0.5 * dim.x;
    const float t = dot(WorldRayDirection(), C - WorldRayOrigin());
    const float l = length(WorldRayOrigin() - C + t * WorldRayDirection());
    if(l >= r) return;
    const float w = sqrt(r * r - l * l);
    const float rayTIn = max(RayTMin(), t - w);
    const float ds = min(/*TMax*/ RayTCurrent(), t + w) - rayTIn;
    if(ds <= 0.0) return;

    BoxIntersectionAttributes attribs;
    attribs.aabbOffset = aabbOffset;
    ReportHit(rayTIn, 0, attribs);
}

[shader("anyhit")]
void boxAnyHit(inout BillboardPayload rayData : SV_RayPayload, BoxIntersectionAttributes attribs : SV_IntersectionAttributes)
{

    // get vertex data, transfer ray differentials and obtain shading data
    BillboardVertex v = getSphericalBillboardVertexData(attribs.aabbOffset, WorldRayOrigin(), WorldRayDirection(), RayTCurrent(), RayTMin(), rayData.rayTMax);
    RayFootprint volumeFootprint = rayData.footprint; // copy footprint and transfer to hit surface
    const AABB bbox = gScene.proceduralPrimitiveAABBs[attribs.aabbOffset];
    volumeFootprint.hitSphere(v.normalW, /*reconstruct original ray origin*/ WorldRayOrigin() + RayTMin() * WorldRayDirection(), WorldRayDirection(), RayTCurrent() - RayTMin(), bbox.center(), 0.5 * bbox.extent().x); // transfer to surface
    BillboardShadingData sd = prepareBillboardShadingData(v, volumeFootprint, attribs.aabbOffset, WorldRayOrigin(), WorldRayDirection());
    //float4 curColor = shadeBillboard(sd, v.billboardDepth);
    float4 maxColor = float4(0);
    float tcurrent = RayTCurrent();
    if(tcurrent > rayData.listMin){
    if(rayData.pos < TRANSPARENT_DEPTH){
        [unroll] for(uint i=0; i< TRANSPARENT_DEPTH; i++){
            if(i==rayData.pos)
            {
                rayData.spheres[i] = attribs.aabbOffset;
                rayData.tcurrent[i] = tcurrent;
                rayData.listMax = max(tcurrent, rayData.listMax);
            }
        }
        rayData.pos= rayData.pos +1;
    }
    else{
        if(tcurrent < rayData.listMax)
        {
            float newListMax = 0;
            [unroll] for(uint i=0; i< TRANSPARENT_DEPTH; i++){
                if(rayData.tcurrent[i]==rayData.listMax)
                {
                    v = getSphericalBillboardVertexData(rayData.spheres[i], WorldRayOrigin(), WorldRayDirection(), RayTCurrent(), RayTMin(), rayData.rayTMax);
                    sd = prepareBillboardShadingData(v, volumeFootprint, rayData.spheres[i], WorldRayOrigin(), WorldRayDirection());
                    maxColor = shadeBillboard(sd, v.billboardDepth);
                    rayData.spheres[i] = attribs.aabbOffset;
                    rayData.tcurrent[i] = tcurrent;
                }
                newListMax = max(rayData.tcurrent[i], newListMax);
            }
            rayData.listMax = newListMax;
        }
        else{
            maxColor = shadeBillboard(sd, v.billboardDepth);
        }
        if(rayData.listMax > rayData.t_leftover)
        {
            rayData.leftover.rgb += maxColor.rgb * rayData.leftover.a;
            rayData.leftover.a *= rayData.leftover.a;
            rayData.t_leftover = rayData.listMax;
        }
        else
        {
            rayData.leftover.rgb = maxColor.rgb + maxColor.a * rayData.leftover.rgb;
            rayData.leftover.a*= maxColor.a;
        }
        

    }
    }

    //if(curColor.a == 0.0) return; // opaque object => abort (hopefully ignores most of the objects behind this one)
    //if(rayData.transmit <= TRANSMIT_EPSILON) return; // this can cause some weird spherical shaped artifacts
    IgnoreHit(); // do not accept the hit in general => keep any-hit traversal going
}

[shader("closesthit")]
void boxClosestHit(inout RayPayload rayData : SV_RayPayload, BoxIntersectionAttributes attribs : SV_IntersectionAttributes)
{
    // this shader is skipped if weighted OIT is used (spherical billboards)
    // store hit information in ray payload
    rayData.depth = RayTCurrent();
    rayData.hit.x = attribs.aabbOffset; // this is the only relevant data for now
}

/****** Ray Kernel for Bilboards ******/


void traceBillboards(inout float3 color, inout float transmit, RayDesc ray, RayFootprint footprint)
{
    // trace single ray and handle everything in any-hit shaders
    BillboardPayload volumePayload;
    volumePayload.footprint = footprint;
    volumePayload.rayTMax = ray.TMax;
    volumePayload.pos = 0;
    volumePayload.listMax = 0;
    volumePayload.listMin = 0;
    volumePayload.leftover = float4(0, 0, 0, 1);
    volumePayload.t_leftover = 0;

    VolumeNode total;
    total.color =0;
    total.transmit =1;

    to_sort sorted[2*TRANSPARENT_DEPTH];
    int index[TRANSPARENT_DEPTH]= {0};
    
    float old = -0.0001;

    TraceRay(gRtScene, RAY_FLAG_SKIP_TRIANGLES | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, 0xff /* instanceInclusionMask */, 0 /* hitIdx */, hitProgramCount, 0 /* missIdx */, ray, volumePayload);
    VolumeNode list[TRANSPARENT_DEPTH];

//    [unroll] for(uint k = 0; k< 1 && volumePayload.listMax != 0; k++){
    for (uint i = 0; i < TRANSPARENT_DEPTH; i++)
  {
    VolumeNode temp;
    temp.depthStart = 0;
    temp.depthEnd = 0;
    temp.transmit = 1;
    temp.color = float3(0);
    list[i] = temp;
  }

      for (uint i = 0; i < volumePayload.pos&&i<TRANSPARENT_DEPTH; i++)
  {
    VolumeNode temp;
    BillboardVertex v =
      getSphericalBillboardVertexData (volumePayload.spheres[i], ray.Origin,
				       ray.Direction,
				       volumePayload.tcurrent[i], ray.TMin,
				       volumePayload.rayTMax);
    RayFootprint volumeFootprint = volumePayload.footprint;	// copy footprint and transfer to hit surface
    const AABB bbox =
      gScene.proceduralPrimitiveAABBs[volumePayload.spheres[i]];
    volumeFootprint.hitSphere (v.normalW, /*reconstruct original ray origin */ ray.Origin + ray.TMin * ray.Direction, ray.Direction, volumePayload.tcurrent[i] - ray.TMin, bbox.center (), 0.5 * bbox.extent ().x);	// transfer to surface
    BillboardShadingData sd =
      prepareBillboardShadingData (v, volumeFootprint,
				   volumePayload.spheres[i], ray.Origin,
				   ray.Direction);
    float4 curColor = shadeBillboard(sd, v.billboardDepth);
     if(abs(curColor.a-1.0) > 0 && abs(v.billboardDepth) > 0)
    {
        temp.transmit = pow(curColor.a, 1.0/(v.billboardDepth));
        //curColor.rgb = curColor.rgb / (1-curColor.a) * (1- temp.transmit);
        temp.color = (curColor.rgb * log(temp.transmit))/(curColor.a-1.0); //(curColor.rgb * log(curColor.a))/(curColor.a-1.0);
        
    }
    else{
        temp.color = float3(0);
        temp.transmit = 1;

    }    
    temp.depthStart = volumePayload.tcurrent[i];
    temp.depthEnd = min(volumePayload.tcurrent[i] +  v.billboardDepth, volumePayload.rayTMax-0.0001);
    list[i] = temp;
    }
    
    



    [unroll] for(uint i =0; i<TRANSPARENT_DEPTH; i++){
        to_sort temp_start, temp_end;
        temp_end.index = i;
        temp_start.index = i;
        temp_start.end = 0;
        temp_end.end = 1;
        temp_start.value = list[i].depthStart;
        temp_end.value = list[i].depthEnd;
        sorted[2*i]=temp_start;
        sorted[2*i+1]= temp_end;
    }
    
    //Bubble sort
    
    uint i, j;
    uint n= 2*TRANSPARENT_DEPTH;
    [unroll] for (i = 0; i < n-1; i++){    
    for (j = 0; j < n-i-1; j++){
        if (sorted[j].value > sorted[j+1].value){
            to_sort temp = sorted[j];
            sorted[j]= sorted[j+1];
            sorted[j+1]= temp;
        }
    }
    }
    
    /*
    //Insertion sort felix bachelorarbeit 
    uint i, j;
    uint n= 2*TRANSPARENT_DEPTH;
    [unroll] for(int i = 1; i<n; ++i){
    for(int j =i; j>0&& sorted[j-1].value > sorted[j].value;--j)
    {
        to_sort tmp = sorted[j];
        sorted[j]=sorted[j-1];
        sorted[j-1] = sorted[j];
    }
    }
    */
    float3 temp_color = float3(0);
    float temp_trans = 1;
    float3 tot_color = float3(0);
    float  tot_trans = 1;
    float3 current_color_values = float3(0);
    float current_transmit_value = 1;
    float prev_pos = -1;
    float3 normed_color;
    uint active = 0;
    [unroll] for(uint i = 0; i<2*TRANSPARENT_DEPTH; i++){
        to_sort temp = sorted[i];
        if(active>0)
        {
            if(prev_pos != -1 && prev_pos != temp.value)
            {
                temp_trans = pow(current_transmit_value, abs(temp.value - prev_pos));
                temp_color =current_color_values * (temp_trans-1) / log(current_transmit_value);
                tot_color = tot_color + (tot_trans * temp_color);
                tot_trans = tot_trans * temp_trans;
                
            /*
                normed_color = (current_color_values*(current_transmit_value-1.0)) / log(current_transmit_value);
                temp_trans = pow(current_transmit_value, abs(temp.value - prev_pos));
                temp_color = normed_color * (1-temp_trans) / current_transmit_value;
                tot_color = tot_color + (tot_trans * temp_color);
                tot_trans = tot_trans * temp_trans;
                
                temp_trans = pow(current_transmit_value, abs(temp.value - prev_pos));
                temp_color = (current_color_values*(temp_trans-1.0)) / log(temp_trans);
                tot_color = tot_color + (tot_trans * temp_color);
                tot_trans = tot_trans * temp_trans;
                */
            }
         }
         else
         {
                current_color_values = float3(0);
                current_transmit_value = 1;
         }
         prev_pos = temp.value;
         if(temp.end == 0)
         {
             active = active +1;
             current_color_values.r = max(current_color_values.r + list[temp.index].color.r, 0);
             current_color_values.g = max(current_color_values.g + list[temp.index].color.g, 0);
             current_color_values.b = max(current_color_values.b + list[temp.index].color.b, 0);
             current_transmit_value =min(current_transmit_value*list[temp.index].transmit, 1);
         }
         if(temp.end == 1)
         {
             active = active -1;
             current_color_values.r = max(current_color_values.r - list[temp.index].color.r, 0);
             current_color_values.g = max(current_color_values.g - list[temp.index].color.g, 0);
             current_color_values.b = max(current_color_values.b - list[temp.index].color.b, 0);
             current_transmit_value = min(current_transmit_value/list[temp.index].transmit, 1);
         }
    }
    //total.color += total.transmit * volumePayload.leftover.rgb;
    //total.transmit *= volumePayload.leftover.a;
    // use weighted-blended OIT to calculate the color
    if( volumePayload.pos > 0)
    {
    uint ind = 2*(7);
    //color = float3(0, list[sorted[ind].index].depthStart, list[sorted[ind].index].depthEnd);//
    //transmit = 0;
    color += transmit *tot_color;//volumePayload.listMax;//float3(total.transmit, list[sorted[volumePayload.pos].index].depthStart, list[sorted[volumePayload.pos].index].depthEnd);//total.color;//float3(list[0].depthEnd, list[0].depthStart, list[0].depthEnd- list[0].depthStart);
    transmit *= tot_trans;// *= tot_trans;
    //color = float3(total.transmit);
    //transmit = 0;
    }
}


/****** Ray Generation ******/
[shader("raygeneration")]
void rayGen()
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 launchDim = DispatchRaysDimensions().xy;
    RayDesc ray = gScene.camera.computeRayPinhole(launchIndex, launchDim).toRayDesc();
    ray.TMin = 0.0; // set to 0 to make handling with ray differentials and origin correction somewhat easier
    RayFootprint footprint = RayFootprint.create(launchIndex, launchDim); // init ray differential

    float3 color = 0.0; // color sum
    float transmit = 1.0; // current visibility

    uint iteration = 0;
    uint internalReflections = 0; // limit this for performance reasons
    bool inVolume = false; // simplified tracking if in volume (glass for our secnes) => only works when not starting inside
    do
    {
        // first trace triangles to determine depth range
        // (only required for spherical and soft particles, could be merged with procedural path otherwise)
        RayPayload trianglePayload;
        trianglePayload.depth = FLT_MAX;
        trianglePayload.hit = HitInfo::kInvalidIndex;
        ray.TMax = FLT_MAX;
        TraceRay(gRtScene, RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES, 0xff /* instanceInclusionMask */, 0 /* hitIdx */, hitProgramCount, 0 /* missIdx */, ray, trianglePayload);

        // handle billboards
        ray.TMax = trianglePayload.depth; // set max range for billboard ray
        if(!inVolume)
            traceBillboards(color, transmit, ray, footprint); // updates color and transmittance based on encountered billboards

        // evaluate triangle geometry (shading was deferred to ray generation shader)
        HitInfo hit;
        if(transmit <= TRANSMIT_EPSILON || !hit.decode(trianglePayload.hit)) break;
      
        // unpack hit data 
        StaticVertexData triangleVertices[3];
        VertexData v = gScene.getVertexData(hit, triangleVertices);
        const uint materialID = gScene.getMaterialID(hit.instanceID);

        // transfer ray differentials and get shading data:
        // compute the travelled distance: When the ray origin correction is used, the ray distance must be subtracted from ray.TMin
        const float travelledDistance = trianglePayload.depth - ray.TMin;
        footprint.hitSurface(v.faceNormalW, travelledDistance, ray.Direction); // transfer ray differentials onto surface
        ShadingData sd = prepareShadingData(v, footprint, triangleVertices, hit, travelledDistance, ray.Direction);
        float4 curColor = shade(sd); // rgb = (pre-multiplied) color, a = transmittance (1 - alpha)

        color += transmit * curColor.rgb;

        // calc fresnel to see how much gets reflected/refracted
        const float3 reflected = reflect(ray.Direction, sd.N); // reflected ray direction
        float3 F = fresnelSchlick(sd.specular, 1, saturate(dot(reflected, sd.N)));
        curColor.a *= (1.0 - dot(F, 0.33)); // adjust fragment transmittance with fresnel
        // prepare next ray
        bool isTransmittive = (curColor.a * transmit > TRANSMIT_EPSILON);
        bool isInternalReflection = sd.eta > 1.0 && sd.eta * sd.eta * (1.0 - dot(sd.N, sd.V) * dot(sd.N, sd.V)) >= 1.0;
        bool isReflective = dot(F, 0.33) * transmit > REFLECT_EPSILON && sd.linearRoughness < 0.02;
        // prepare correct internal reflection
        if(isInternalReflection && internalReflections < MAX_INTERNAL_REFLECTIONS){
            ++internalReflections;
            isTransmittive = false;
            isReflective = transmit > REFLECT_EPSILON && sd.linearRoughness < 0.02;
            F = 1.0;
        }
        // compute ray length for origin correction
        float correctionEta = 1.0;
        if(isTransmittive) correctionEta = 1.0 / sd.eta;
        float correctionDepth = trianglePayload.depth * correctionEta;

        if(isTransmittive) // prefer transmission over reflections
        {
            if(isReflective) // envmap lookup for reflection
                color += F * transmit * gScene.envMap.eval(reflected);

            inVolume = sd.eta < 1.0;
            transmit *= curColor.a; // update visibility
            float3 refracted = ray.Direction;
            if(sd.IoR != 1.0 && !isInternalReflection) // workaround for internal reflection that looks good and is not too expensive (avoids back and forth bounces)
                refracted = refract(ray.Direction, sd.N, sd.eta);

            // update ray differentials
            footprint.bounceOnSurface(v, triangleVertices, hit, sd.N, ray.Direction, refracted, true);
            ray.Direction = refracted;
            /*if(USE_REFRACTION_CORRECTION) { // apply ray origin correction for billboards
                ray.Origin = sd.computeNewRayOrigin(false) - refracted * correctionDepth;
                ray.TMin = correctionDepth;
            } else { */// cast new ray from intersection point
                ray.Origin = sd.computeNewRayOrigin(false);
                ray.TMin = 0.0;
            /*}*/
        }
        else if(isReflective) // reflect ray
        {
            transmit *= dot(F, 0.33); // update visibility
            // update ray differentials
            footprint.bounceOnSurface(v, triangleVertices, hit, sd.N, ray.Direction, reflected, false);
            ray.Direction = reflected;
            /*if(USE_REFLECTION_CORRECTION) { // apply ray origin correction for billboards
                ray.Origin = sd.computeNewRayOrigin() - reflected * correctionDepth;
                ray.TMin = correctionDepth;
            } else { */// cast new ray from intersection point
                ray.Origin = sd.computeNewRayOrigin();
                ray.TMin = 0.0;
            /*}*/
        }
        else transmit = 0.0; // stop tracing rays
    } while(++iteration < MAX_BOUNCES && transmit > TRANSMIT_EPSILON);

    if(transmit > TRANSMIT_EPSILON) // sample environment map (miss)
    {
        color += transmit * gScene.envMap.eval(ray.Direction);
    }
    
    gOutputColor[launchIndex] = float4(color, 1);
}
