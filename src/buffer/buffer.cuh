#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <memory>
#include <cstdint>

#include <cuda_runtime.h>

namespace gph
{

template <typename T>
using Ptr = std::shared_ptr<T>;

void initGraphite();
void destroyGraphite();
void draw();

class FrameBuffer;
class VertexBuffer;
class IndexBuffer;

//------------------//
//  BufferRegister  //
//------------------//

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

    BufferRegister(const BufferRegister& bufferRegister) = delete;
    BufferRegister(BufferRegister&& bufferRegister) noexcept = delete;

    BufferRegister& operator=(const BufferRegister& bufferRegister) = delete;
    BufferRegister& operator=(BufferRegister&& bufferRegister) noexcept = delete;

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

//-------------//
//   Buffer    //
//-------------//

class Buffer {
protected:
    void* buffer;
    unsigned int id;
    size_t size;
public:
    Buffer(unsigned int _id, size_t _size);
    Buffer() = default;

    virtual ~Buffer();
    
    Buffer(const Buffer& buff);
    Buffer(Buffer&& buff) noexcept;

    Buffer& operator=(const Buffer& buff);
    Buffer& operator=(Buffer&& buff) noexcept;
public:
    virtual void bind() = 0;
    virtual void unbind() = 0;
public:
    inline void* getBuffer() const { return buffer; }
    inline unsigned int getID() const { return id; }
    inline size_t getSize() const { return size; }
};

//-----------------//
//   FrameBuffer   //
//-----------------//

class FrameBuffer : public Buffer {
private:
    unsigned int width, height;
public:
    FrameBuffer(unsigned int id, unsigned int _width, unsigned int _height);
    FrameBuffer(unsigned int _width, unsigned int _height);
    FrameBuffer() = default;

    ~FrameBuffer() = default;
    
    FrameBuffer(const FrameBuffer& frameBuffer);
    FrameBuffer(FrameBuffer&& frameBuffer) noexcept;

    FrameBuffer& operator=(const FrameBuffer& frameBuffer);
    FrameBuffer& operator=(FrameBuffer&& frameBuffer) noexcept;
public:
    static Ptr<FrameBuffer> New(unsigned int width, unsigned int height);
public:
   void bind() override;
   void unbind() override;
public:
    inline unsigned int getWidth() const { return width; }
    inline unsigned int getHeight() const { return height; }
};

//------------------//
//   VertexBuffer   //
//------------------//

class VertexBuffer : public Buffer {
public:
    VertexBuffer(unsigned int id, float* data, size_t size);
    VertexBuffer() = default;

    ~VertexBuffer() = default;

    VertexBuffer(const VertexBuffer& vertexBuffer);
    VertexBuffer(VertexBuffer&& vertexBuffer) noexcept;

    VertexBuffer& operator=(const VertexBuffer& vertexBuffer);
    VertexBuffer& operator=(VertexBuffer&& vertexBuffer) noexcept;
public:
    static Ptr<VertexBuffer> New(float* data, size_t size);
public:
    void bind() override;
    void unbind() override;
};

//-----------------//
//   IndexBuffer   //
//-----------------//

class IndexBuffer : public Buffer {
public:
    IndexBuffer(unsigned int id, unsigned int* indices, size_t size);
    IndexBuffer() = default;

    ~IndexBuffer() = default;

    IndexBuffer(const IndexBuffer& indexBuffer);
    IndexBuffer(IndexBuffer&& indexBuffer) noexcept;

    IndexBuffer& operator=(const IndexBuffer& indexBuffer);
    IndexBuffer& operator=(IndexBuffer&& indexBuffer) noexcept;
public:
    static Ptr<IndexBuffer> New(unsigned int* indices, size_t size);
public:
    void bind() override;
    void unbind() override;
};

//------------//
//    CUDA    //
//------------//

void check_cuda_error(const char* message);

}
