add_mlir_translation_library(TritonAMDGCN
        AMDGCNTranslation.cpp

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        TritonLLVMIR
        )
