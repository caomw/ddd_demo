
#ifndef __PC2TSDF_H_
#define __PC2TSDF_H_

#include "ext/common.h"
#include "ext/utility.h"
#include "ext/stringUtil.h"
#include "ext/vec3.h"
#include "ext/vec4.h"
#include "ext/mat4.h"
#include "ext/grid2.h"
#include "ext/grid3.h"
#include "ext/boundingBox3.h"

#include "ext/plyHeader.h"
#include "ext/pointCloud.h"
#include "ext/pointCloudIO.h"
#include "ext/uniformAccelerator.h"

using namespace ml;

namespace pc2tsdf
{

struct TSDFHeader
{
    int headerSize;

    int dimX;
    int dimY;
    int dimZ;

    vec3f worldOrigin;
    float voxelSize;
    float truncationRadius;
};

struct TSDF
{
    vec3f getVoxelCenter(const vec3ui &cell) const
    {
        return worldOrigin + vec3f(cell) * voxelSize + vec3f(voxelSize * 0.5f);
    }

    void saveBinary(const std::string &filename) const
    {
        FILE *file = util::checkedFOpen(filename.c_str(), "wb");
        if (!file)
        {
            std::cout << "Failed to open file: " << file << std::endl;
            return;
        }

        TSDFHeader header;
        header.headerSize = sizeof(TSDFHeader);
        header.dimX = (int)data.getDimensions().x;
        header.dimY = (int)data.getDimensions().y;
        header.dimZ = (int)data.getDimensions().z;

        header.worldOrigin = worldOrigin;
        header.voxelSize = voxelSize;
        header.truncationRadius = truncationRadius;

        util::checkedFWrite(&header, sizeof(header), 1, file);

        util::checkedFWrite(data.getData(), sizeof(float), header.dimX * header.dimY * header.dimZ, file);

        fclose(file);
    }

    vec3f worldOrigin;
    float voxelSize;
    float truncationRadius;

    Grid3f data;
};

class PointCloudToTSDF
{
public:
    void makeTSDF(const PointCloudf &cloud, float voxelSize, float truncationRadius, TSDF &out);

private:
    float computeTSDFValue(const PointCloudf &cloud, const vec3f &pos, float truncationRadius);
    
    UniformAccelerator accel;
};

inline void makeTSDF(const PointCloudf &cloud, float voxelSize, float truncationRadius, TSDF &out)
{
    PointCloudToTSDF maker;
    maker.makeTSDF(cloud, voxelSize, truncationRadius, out);
}

}

#include "ext/pc2tsdf.inl"

#endif