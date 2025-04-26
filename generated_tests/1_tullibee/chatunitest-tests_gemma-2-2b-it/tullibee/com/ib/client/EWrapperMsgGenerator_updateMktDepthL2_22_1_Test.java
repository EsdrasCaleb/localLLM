package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

public class EWrapperMsgGenerator_updateMktDepthL2_22_1_Test {

    @Test
    void updateMktDepthL2() {
        EWrapperMsgGenerator.updateMktDepthL2(1, 1, "marketMaker", 1, 1, 1.0, 1);
    }
}
