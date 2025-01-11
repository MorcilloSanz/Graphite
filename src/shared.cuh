#include <iostream>
#include <memory>

#define SHARED_PTR(clazz) public: \
                            using Ptr = std::shared_ptr<clazz>; \
                            template<class... Args> \
                            inline static Ptr New(Args&&... args) { \
                                return std::make_shared<clazz>(std::forward<Args>(args)...); \
                            }\

