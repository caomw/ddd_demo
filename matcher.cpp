#include "pc2tsdf/pc2tsdf.h"
#include "detect_keypoints.h"
#include "ddd.h"
#include "fragmentMatcher/fragmentMatcher.h"

using namespace std;

int main()
{
    const float voxelSize = 0.01f;
    const float truncationRadius = 0.05f;
    const float maxKeypointMatchDist = 0.02f;
    const int fragmentCount = 57;
    const string fragmentPrefix = "/data/andyz/kinfu/data/augICLNUIMDataset/fragments/livingroom1-fragments-ply/cloud_bin_";

    vector<string> allFragments;
    
    for (int i = 0; i < fragmentCount; i++)
    {
        const string fragmentFilename = fragmentPrefix + to_string(i) + ".ply";
        assert(util::fileExists(fragmentFilename));
        allFragments.push_back(fragmentFilename);
    }
    
    for (int i = 0; i < fragmentCount; i++)
    {
        for (int j = 0; j < fragmentCount; j++)
        {
            const string resultFilename = "results/match" + to_string(i) + "-" + to_string(j) + ".txt";
            if (i <= j || util::fileExists(resultFilename))
                continue;
            
            auto result = FragmentMatcher::match(allFragments[i], allFragments[j], i, j, voxelSize, truncationRadius, maxKeypointMatchDist);
            result.saveASCII(resultFilename);
        }
    }

    return 0;
}
