#ifndef bitmap_hpp
#define bitmap_hpp
#include <stdio.h>
#include <vector>
#include <cstring>
#include <cinttypes>
#include <iostream>
#include <fstream>
/**
 * @class Bitmap
 * @brief writes a bitmap out to a file
 */
class Bitmap {
private:
    /**
     * @brief get four byes
     * @param data data element
     * @return vector of 4 bytes
     */
    static std::vector<uint8_t> get_4_bytes(const uint32_t & data);
    /**
     * @brief insert 4 bytes into the destination vector
     * @param dest destination vector
     * @param data data to pack into vector
     */
    static void insert_4_bytes(std::vector<uint8_t> & dest, const uint32_t & data);
    /**
     * @brief get two bytes
     * @param data element from which to get two bytes
     * @return vector of 2 bytes
     */
    static std::vector<uint8_t> get_2_bytes(const uint16_t & data);
    /**
     * @brief insert 2 bytes into destination vector
     * @param dest destination vector
     * @param data data to pack into destination vector
     */
    static void insert_2_bytes(std::vector<uint8_t> & dest, const uint16_t & data);
    /**
     * @brief encode rgd data into a bitmap image
     * @param rgb rgb data
     * @param width width of image
     * @param height height of image
     * @return output output 2d vector of bytes
     */
    static size_t bitmap_encode_rgb(const uint8_t* rgb, int width, int height, uint8_t** output);
public:
    /**
     * @brief write bitmap file
     * @param bitmapData bitmap data
     * @param w width
     * @param h height
     * @param binary whether or not the data are binary valued or 0-255 integers
     * @param fileName name of file to write to
     */
    static void writeBitmap(std::vector<unsigned char> bitmapData, int w, int h, std::string fileName, bool binary=true);
};
#endif
