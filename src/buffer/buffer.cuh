#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <cstdint>

#include <cuda_runtime.h>

namespace ghp
{

template <typename T>
using Ptr = std::shared_ptr<T>;

void initGraphite();
void destroyGraphite();
void draw();

class FrameBuffer;
class VertexBuffer;
class IndexBuffer;

class BufferRegister {
protected:
    std::vector<Ptr<FrameBuffer>> fbos;
    std::vector<Ptr<VertexBuffer>> vbos;
    std::vector<Ptr<IndexBuffer>> ibos;

    int fboID, vboID, iboID;

    static BufferRegister* instance;
protected:
    BufferRegister() 
        : fboID(0), vboID(0), iboID(0) {
    }
public:
    ~BufferRegister() = default;
    static BufferRegister* getInstance();
    static void destroyInstance();
public:
    inline void addFrameBuffer(const Ptr<FrameBuffer>& fbo) { fbos.push_back(fbo); }
    inline void addVertexBuffer(const Ptr<VertexBuffer>& vbo) { vbos.push_back(vbo); }
    inline void addIndexBuffer(const Ptr<IndexBuffer>& ibo) { ibos.push_back(ibo); }

    inline const std::vector<Ptr<FrameBuffer>>& getFrameBuffers() { return fbos; }
    inline const std::vector<Ptr<VertexBuffer>>& getVertexBuffers() { return vbos; }
    inline const std::vector<Ptr<IndexBuffer>>& getIndexBuffers() { return ibos; }

    inline Ptr<FrameBuffer> getBindedFrameBuffer() const { return fbos[fboID - 1]; }
    inline Ptr<VertexBuffer> getBindedVertexBuffer() const { return vbos[vboID - 1]; }
    inline Ptr<IndexBuffer> getBindedIndexBuffer() const { return ibos[iboID - 1]; }

    inline int getBindedFrameBufferID() const { return fboID; }
    inline int getBindedVertexBufferID() const { return vboID; }
    inline int getBindedIndexBufferID() const { return iboID; }

    inline void bindFbo(int fboID) { this->fboID = fboID; }
    inline void bindVbo(int vboID) { this->vboID = vboID; }
    inline void bindIbo(int iboID) { this->iboID = iboID; }
};

class Buffer {
public:
    virtual void bind() = 0;
    virtual void unbind() = 0;
};

class FrameBuffer : public Buffer {
private:
    int id;
    uint8_t* buffer;
    unsigned int width, height;
public:
    FrameBuffer(int _id, unsigned int _width, unsigned int _height);
    FrameBuffer(unsigned int _width, unsigned int _height);
    ~FrameBuffer();
    FrameBuffer() = default;
public:
    static Ptr<FrameBuffer> New(unsigned int width, unsigned int height);
public:
   void bind() override;
   void unbind() override;

   void draw();
public:
    inline int getID() const { return id; }
    inline uint8_t* getBuffer() { return buffer; }
    
    inline unsigned int getWidth() const { return width; }
    inline unsigned int getHeight() const { return height; }

    inline size_t size() const { return width * height * 3; }
};

void check_cuda_error(const char* message);

}
