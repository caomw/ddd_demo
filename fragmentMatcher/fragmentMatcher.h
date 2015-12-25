
#include "pc2tsdf.h"

class FragmentMatcher
{
public:
    struct KeypointMatch
    {
        vec3f posA;
        vec3f posB;
        float alignmentError;
    };

    struct Result
    {
        bool matchFound;

        mat4f transformBToA;
        
        std::vector<vec3f> keypointsA;
        std::vector<vec3f> keypointsB;
        std::vector<KeypointMatch> matches;
    };

    static Result match(const std::string &pointCloudFileA, const std::string &pointCloudFileB, float voxelSize, float truncationRadius, float maxKeypointMatchDist)
    {
        std::cout << "Matching " << pointCloudFileA << " against " << pointCloudFileB << std::endl;

        FlatTSDF tsdfA = plyToTSDF(pointCloudFileA, voxelSize, truncationRadius);
        FlatTSDF tsdfB = plyToTSDF(pointCloudFileB, voxelSize, truncationRadius);

        std::cout << "OriginA: " << tsdfA.origin << std::endl;
        std::cout << "DimA: " << tsdfA.dim << std::endl;

        ///////////////////////////////////////////////////////////////////

        const float k_match_score_thresh = 0.5f;
        const float ransac_k = 10; // RANSAC over top-k > k_match_score_thresh
        const float max_ransac_iter = 1000000;
        const float ransac_inlier_thresh = 0.01f;
        
        float* Rt = new float[12]; // Contains rigid transform matrix
        align2tsdf(
            tsdfA.data.data(), tsdfA.dim.x, tsdfA.dim.y, tsdfA.dim.z, tsdfA.origin.x, tsdfA.origin.y, tsdfA.origin.z,
            tsdfB.data.data(), tsdfB.dim.x, tsdfB.dim.y, tsdfB.dim.z, tsdfB.origin.x, tsdfB.origin.y, tsdfB.origin.z,
            voxelSize, k_match_score_thresh, ransac_k, max_ransac_iter, ransac_inlier_thresh, Rt);

        ///////////////////////////////////////////////////////////////////

        // TODO: this is redundant with align2tsdf
        Result result;
        result.keypointsA = tsdfA.makeKeypoints();
        result.keypointsB = tsdfB.makeKeypoints();

        result.transformBToA = mat4f::identity();
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 3; j++)
                result.transformBToA(j, i) = Rt[j * 4 + i];
        
        pc2tsdf::UniformAccelerator acceleratorA(result.keypointsA, maxKeypointMatchDist);
        
        //transform B's keypoints into A's
        std::vector<vec3f> keypointsBtransformed = result.keypointsB;
        for (auto &bPt : keypointsBtransformed)
        {
            const vec3f bPtInA = result.transformBToA * bPt;
            const auto closestPt = acceleratorA.findClosestPoint(bPtInA);
            const float dist = closestPt.second;
            if (dist <= maxKeypointMatchDist)
            {
                KeypointMatch match;
                match.posA = closestPt.first;
                match.posB = bPt;
                match.alignmentError = dist;
                result.matches.push_back(match);
            }
        }

        result.matchFound = result.matches.size() > 0;

        std::cout << "Keypoint matches found: " << result.matches.size() << std::endl;

        return result;
    }

private:
    
    struct FlatTSDF
    {
        std::vector<vec3f> makeKeypoints()
        {
            auto keypointsInt = detect_keypoints_filtered(data.data(), dim.x, dim.y, dim.z);
            std::vector<vec3f> result;
            for (auto &x : keypointsInt)
                result.push_back(vec3f(x[0], x[1], x[2]) * voxelSize + origin);
            return result;
        }
        vec3i dim;
        vec3f origin;
        float voxelSize;
        std::vector<float> data;
    };

    static FlatTSDF plyToTSDF(const std::string &filename, float voxelSize, float truncationRadius)
    {
        auto cloud = PointCloudIOf::loadFromFile(filename);
        std::cout << "Cloud size: " << cloud.m_points.size() << ", p5: " << cloud.m_points[5] << std::endl;
        pc2tsdf::TSDF tsdf;
        pc2tsdf::makeTSDF(cloud, voxelSize, truncationRadius, tsdf);
        
        // Convert TUDF grid data to float array
        FlatTSDF result;
        const vec3i dim = tsdf.data.getDimensions();
        result.dim = dim;
        result.data = std::vector<float>(dim.x * dim.y * dim.z);
        for (float &x : result.data)
            x = 1.0f;
        result.origin = tsdf.worldOrigin;
        result.voxelSize = voxelSize;
        
        for (int z = 0; z < dim.z; z++)
            for (int y = 0; y < dim.y; y++)
                for (int x = 0; x < dim.x; x++)
                    result.data[z * dim.y * dim.x + y * dim.x + x] = tsdf.data(x, y, z) / truncationRadius;
        return result;
    }
};