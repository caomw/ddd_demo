#pragma once



#ifndef _POINT_CLOUD_IO_H_
#define _POINT_CLOUD_IO_H_

namespace ml {

template <class FloatType>
class PointCloudIO {
public:

	static PointCloud<FloatType> loadFromFile(const std::string& filename) {
		PointCloud<FloatType> pc;
		loadFromFile(filename, pc);
		return pc;
	}

	static void loadFromFile(const std::string& filename, PointCloud<FloatType>& pointCloud) {
		pointCloud.clear();
		std::string extension = util::getFileExtension(filename);

		if (extension == "ply") {
			loadFromPLY(filename, pointCloud);
		} else {
			//throw MLIB_EXCEPTION("unknown file extension" + filename);
		}

		//if (!pointCloud.isConsistent()) throw MLIB_EXCEPTION("inconsistent point cloud");
	}


	static void saveToFile(const std::string& filename, const std::vector<vec3<FloatType>> &points) {
		PointCloud<FloatType> pc;
		pc.m_points = points;
		saveToFile(filename, pc);
	}

	static void saveToFile(const std::string& filename, const PointCloud<FloatType>& pointCloud) {
		if (pointCloud.isEmpty()) {
			MLIB_WARNING("empty point cloud");
			return;
		}
		std::string extension = util::getFileExtension(filename);
		if (extension == "ply") {
			writeToPLY(filename, pointCloud);
		} else {
			//throw MLIB_EXCEPTION("unknown file extension" + filename);
		}
	}


	/************************************************************************/
	/* Read Functions													    */
	/************************************************************************/

	static void loadFromPLY(const std::string& filename, PointCloud<FloatType>& pc);


	/************************************************************************/
	/* Write Functions													    */
	/************************************************************************/

	static void writeToPLY(const std::string& filename, const PointCloud<FloatType>& pc) {

		//if (!std::is_same<FloatType, float>::value) throw MLIB_EXCEPTION("only implemented for float, not for double");

		std::ofstream file(filename, std::ios::binary);
		//if (!file.is_open()) throw MLIB_EXCEPTION("Could not open file for writing " + filename);
		file << "ply\n";
		file << "format binary_little_endian 1.0\n";
		file << "comment MLIB generated\n";
		file << "element vertex " << pc.m_points.size() << "\n";
		file << "property float x\n";
		file << "property float y\n";
		file << "property float z\n";
		if (pc.m_normals.size() > 0) {
			file << "property float nx\n";
			file << "property float ny\n";
			file << "property float nz\n";
		}
		if (pc.m_colors.size() > 0) {
			file << "property uchar red\n";
			file << "property uchar green\n";
			file << "property uchar blue\n";
			file << "property uchar alpha\n";
		}
		file << "end_header\n";

		if (pc.m_colors.size() > 0 || pc.m_normals.size() > 0) {
			size_t vertexByteSize = sizeof(float)*3;
			if (pc.m_normals.size() > 0)	vertexByteSize += sizeof(float)*3;
			if (pc.m_colors.size() > 0)		vertexByteSize += sizeof(unsigned char)*4;
			BYTE* data = new BYTE[vertexByteSize*pc.m_points.size()];
			size_t byteOffset = 0;
			for (size_t i = 0; i < pc.m_points.size(); i++) {
				memcpy(&data[byteOffset], &pc.m_points[i], sizeof(float)*3);
				byteOffset += sizeof(float)*3;
				if (pc.m_normals.size() > 0) {
					memcpy(&data[byteOffset], &pc.m_normals[i], sizeof(float)*3);
					byteOffset += sizeof(float)*3;
				}
				// if (pc.m_colors.size() > 0) {
				// 	vec4uc c(pc.m_colors[i]*255);
				// 	memcpy(&data[byteOffset], &c, sizeof(unsigned char)*4);
				// 	byteOffset += sizeof(unsigned char)*4;
				// }
			}
			file.write((const char*)data, byteOffset);
			SAFE_DELETE_ARRAY(data);
		} else {
			file.write((const char*)&pc.m_points[0], sizeof(float)*3*pc.m_points.size());
		}

		file.close();
	}


};

typedef PointCloudIO<float> PointCloudIOf;
typedef PointCloudIO<double> PointCloudIOd;

} //namespace ml


#include "pointCloudIO.inl"

#endif
