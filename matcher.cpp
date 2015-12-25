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
    const string fragmentPrefix = "../ddd_data/cloud_bin_";

    vector<string> allFragments;
    
    for (int i = 0; i < fragmentCount; i++)
    {
        const string fragmentFilename = fragmentPrefix + to_string(i) + ".ply";
        assert(util::fileExists(fragmentFilename));
        allFragments.push_back(fragmentFilename);
    }
    
    ofstream file("dump.txt");
    for (int i = 0; i < fragmentCount; i++)
    {
        for (int j = 0; j < fragmentCount; j++)
        {
            if (i == 0 && j >= 1)
            {
                auto result = FragmentMatcher::match(allFragments[i], allFragments[j], voxelSize, truncationRadius, maxKeypointMatchDist);
                file << result.matches.size() << endl;
            }
        }
    }

    return 0;
}
