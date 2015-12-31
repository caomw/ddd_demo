
#include "pc2tsdf.h"

class DenseCheck
{
public:
    struct Result
    {
        Result()
        {
            commonPointCount = 0;
            overlapRatio = 0.0f;
            totalAvgResidual = 0.0f;
            overlapAvgResidual = 0.0f;
        }
        int commonPointCount;
        float overlapRatio;
        float totalAvgResidual;
        float overlapAvgResidual;
    };

    static Result run(const std::string &pointCloudFileA, const std::string &pointCloudFileB, const mat4f &transformBToA, float matchDist)
    {
        auto cloudA = PointCloudIOf::loadFromFile(pointCloudFileA);
        auto cloudB = PointCloudIOf::loadFromFile(pointCloudFileB);
        return run(cloudA.m_points, cloudB.m_points, transformBToA, matchDist);
    }

    static Result run(const std::vector<vec3f> &pointCloudA, const std::vector<vec3f> &pointCloudB, const mat4f &transformBToA, float matchDist)
    {
        if (pointCloudA.size() < pointCloudB.size())
        {
            return run(pointCloudB, pointCloudA, transformBToA.getInverse(), matchDist);
        }

        const float matchDistSq = matchDist * matchDist;

        UniformAccelerator accelA(pointCloudA, matchDist);

        Result result;

        for (const vec3f &bPt : pointCloudB)
        {
            const vec3f aPt = transformBToA * bPt;
            auto closestAPt = accelA.findClosestPoint(aPt);
            const float distSq = vec3f::distSq(closestAPt.second, aPt);

            if (distSq <= matchDistSq)
            {
                const float dist = sqrtf(distSq);
                result.commonPointCount++;
                result.totalAvgResidual += dist;
                result.overlapAvgResidual += dist;
            }
            else
            {
                result.totalAvgResidual += matchDist;
            }
        }

        if (result.commonPointCount == 0)
        {
            return Result();
        }

        result.overlapRatio = (float)result.commonPointCount / (float)math::min(pointCloudA.size(), pointCloudB.size());
        result.totalAvgResidual /= (float)pointCloudB.size();
        result.overlapAvgResidual /= (float)result.commonPointCount;
        return result;
    }
};