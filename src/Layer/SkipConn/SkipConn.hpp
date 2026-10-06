#pragma once

struct SkipConn {
  public:
    inline bool IsDefined() const { return defined; }
    inline bool IsBuilt() const { return built; }

  private:
    bool defined = false;
    bool built   = false;
};
