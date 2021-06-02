"#version 450

  int idiv(int a, int b, float sign) {
    int res = a / b;
    int mod = a % b;
    if (sign < 0. && mod != 0) {
      res -= 1;
    }
    return res;
  }

  // Checks whether coordinates lie within the bounds of the shape.
  bool coordsInBounds(ivec4 coord, ivec4 shape) {
    return all(greaterThanEqual(coord, ivec4(0))) &&
        all(lessThan(coord, shape));
  }

  bool coordsInBounds(ivec3 coord, ivec3 shape) {
    return all(greaterThanEqual(coord, ivec3(0))) &&
        all(lessThan(coord, shape));
  }

  bool coordsInBounds(ivec2 coord, ivec2 shape) {
    return all(greaterThanEqual(coord, ivec2(0))) &&
        all(lessThan(coord, shape));
  }


      layout (local_size_x = 8,
              local_size_y = 8,
              local_size_z = 1) in;
    

    layout(std430, set = 0, binding = 0) writeonly buffer ssbOut {
      float result[];
    };
  

      layout(std430, set = 0, binding = 1) readonly buffer ssbA {
        float A[];
      };
    

      layout(std430, set = 0, binding = 2) readonly buffer ssbB {
        float B[];
      };
    

        layout(std140, set = 0, binding = 3) uniform Uniforms {
            float NAN; ivec3 aShape; ivec3 bShape; ivec3 outShape; ivec2 outShapeStrides; 
        };
    

      bool isnan_custom(float val) {
        return (val > 0.0 || val < 0.0) ? false : val != 0.0;
      }

      bvec4 isnan_custom(vec4 val) {
        return bvec4(isnan_custom(val.x),
          isnan_custom(val.y), isnan_custom(val.z), isnan_custom(val.w));
      }

      #define isnan(value) isnan_custom(value)
    

  int getFlatIndex(int coord, int shape) {
    return coord;
  }

  int getFlatIndex(ivec2 coords, ivec2 shape) {
    return int(dot(coords, ivec2(shape.y, 1.)));
  }

  int getFlatIndex(ivec3 coords, ivec3 shape) {
    return int(dot(coords, ivec3(shape.y * shape.z, shape.z, 1.)));
  }

  int getFlatIndex(ivec4 coords, ivec4 shape) {
    return int(dot(coords, ivec4(
      shape.y * shape.z * shape.w, shape.z * shape.w, shape.w, 1.)));
  }


    ivec3 getCoordsFromFlatIndex(int index) {
      int d0 = index / outShapeStrides[0]; index -= d0 * outShapeStrides[0];int d1 = index / outShapeStrides[1]; int d2 = index - d1 * outShapeStrides[1];
      return ivec3(d0,d1,d2);
    }
  
ivec3 getOutputCoords() {
    int d2 =
        int(gl_GlobalInvocationID[0]);int d1 =
        int(gl_GlobalInvocationID[1]);int d0 =
        int(gl_GlobalInvocationID[2]);
  return ivec3(d0,d1,d2);}
void setOutput(int flatIndex, float value) {
      result[flatIndex] = value;
    }
    void setOutput(int flatIndex, int value) {
      result[flatIndex] = float(value);
    }
        int getOutputFlatIndex(ivec3 coords) {
          return int(dot(coords, ivec3(outShapeStrides.x, outShapeStrides.y, 1)));
        }
        
      void setOutput(int d0, int d1, int d2, float value) {
        int flatIndex = getOutputFlatIndex(ivec3(d0, d1, d2));
        setOutput(flatIndex, value);
      }
      void setOutput(int d0, int d1, int d2, int value) {
        int flatIndex = getOutputFlatIndex(ivec3(d0, d1, d2));
        setOutput(flatIndex, value);
      }
    

    float getA(int d0, int d1, int d2) {
      return float(A[getFlatIndex(ivec3(d0,d1,d2),
        aShape)]);
    }
   
    float getAAtOutCoords() {
      ivec3 coords = getOutputCoords();
      
      return float(A[getFlatIndex(ivec3(coords[0], coords[1], coords[2]), aShape)]);
    }

    float getAAtOutCoords(ivec3 coords) {
      
      return float(A[getFlatIndex(ivec3(coords[0], coords[1], coords[2]), aShape)]);
    }
  

    float getB(int d0, int d1, int d2) {
      return float(B[getFlatIndex(ivec3(d0,d1,d2),
        bShape)]);
    }
   
    float getBAtOutCoords() {
      ivec3 coords = getOutputCoords();
      
      return float(B[getFlatIndex(ivec3(coords[0], coords[1], coords[2]), bShape)]);
    }

    float getBAtOutCoords(ivec3 coords) {
      
      return float(B[getFlatIndex(ivec3(coords[0], coords[1], coords[2]), bShape)]);
    }
  

      

      int dimAOuter = aShape[1];
      int dimInner = aShape[2];
      int dimBOuter = bShape[2];

      int batch;

      
    float mm_readA(int row, int col);
    float mm_readB(int row, int col);
    void mm_write(int row, int col, float value);
    void mm_matMul(int dimAOuter, int dimInner, int dimBOuter);

    const int RowPerThread = 4;
    const int ColPerThread = 4;
    const int TileAOuter = int(gl_WorkGroupSize.y) * RowPerThread;
    const int TileBOuter = int(gl_WorkGroupSize.x) * ColPerThread;
    const int TileInner = TileAOuter > TileBOuter ? TileAOuter : TileBOuter;

    shared float mm_Asub[TileAOuter][TileInner];
    shared float mm_Bsub[TileInner][TileBOuter];

    void mm_matMul(int dimAOuter, int dimInner, int dimBOuter) {
      int tileRow = int(gl_LocalInvocationID.y) * RowPerThread;
      int tileCol = int(gl_LocalInvocationID.x) * ColPerThread;

      int globalRow = int(gl_GlobalInvocationID.y) * RowPerThread;
      int globalCol = int(gl_GlobalInvocationID.x) * ColPerThread;

      int numTiles = (dimInner - 1) / TileInner + 1;

      float acc[RowPerThread][ColPerThread];
      float ACached;
      float BCached[ColPerThread];

      // Without this initialization strange values show up in acc.
      for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
        for (int innerCol = 0; innerCol < ColPerThread; innerCol++) {
          acc[innerRow][innerCol] = 0.0;
        }
      }

      const int ColPerThreadA = TileInner / int(gl_WorkGroupSize.x);
      int tileColA = int(gl_LocalInvocationID.x) * ColPerThreadA;
      const int RowPerThreadB = TileInner / int(gl_WorkGroupSize.y);
      int tileRowB = int(gl_LocalInvocationID.y) * RowPerThreadB;

      // Loop over shared dimension.
      for (int t = 0; t < numTiles; t++) {
        // Load one tile of A into local memory.
        for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
          for (int innerCol = 0; innerCol < ColPerThreadA; innerCol++) {
            int inputRow = tileRow + innerRow;
            int inputCol = tileColA + innerCol;

            mm_Asub[inputRow][inputCol] = mm_readA(
                globalRow + innerRow,
                t * TileInner + inputCol);
          }
        }
        // Load one tile of B into local memory.
        for (int innerRow = 0; innerRow < RowPerThreadB; innerRow++) {
          for (int innerCol = 0; innerCol < ColPerThread; innerCol++) {
            int inputRow = tileRowB + innerRow;
            int inputCol = tileCol + innerCol;

            mm_Bsub[inputRow][inputCol] = mm_readB(
              t * TileInner + inputRow,
              globalCol + innerCol);;
          }
        }

        barrier();

        // Compute acc values for a single thread.
        for (int k = 0; k < TileInner; k++) {
          for (int inner = 0; inner < ColPerThread; inner++) {
            BCached[inner] = mm_Bsub[k][tileCol + inner];
          }

          for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
            ACached = mm_Asub[tileRow + innerRow][k];
            for (int innerCol = 0; innerCol < ColPerThread; innerCol++) {
              acc[innerRow][innerCol] += ACached * BCached[innerCol];
            }
          }
        }

        barrier();
      }

      for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
        for (int innerCol = 0; innerCol < ColPerThread; innerCol++) {

          if ((globalCol + innerCol) < dimBOuter &&
              (globalRow + innerRow) < dimAOuter) {
            mm_write(globalRow + innerRow,
                     globalCol + innerCol,
                     acc[innerRow][innerCol]);
          }
        }
      }
    }
  
      float mm_readA(int row, int col) {
        int batchASize = aShape[1] * aShape[2];
        return coordsInBounds(ivec2(row, col), ivec2(dimAOuter, dimInner)) ?
            A[batch * batchASize + row * dimInner + col] : 0;
      }
      float mm_readB(int row, int col) {
        int batchBSize = bShape[1] * bShape[2];
        return coordsInBounds(ivec2(row, col), ivec2(dimInner, dimBOuter)) ?
            B[batch * batchBSize + row * dimBOuter + col] : 0;
      }
      void mm_write(int row, int col, float value) {
        ivec3 outCoord = ivec3(batch, row, col);
        
        
        setOutput(batch, row, col, value);
      }
      void main() {
        batch = int(gl_GlobalInvocationID.z);
        mm_matMul(dimAOuter, dimInner, dimBOuter);
      }
    "
