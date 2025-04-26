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

class EWrapperMsgGenerator_scannerParameters_28_0_Test {

    @Test
    void scannerParametersTest() {
        String xml = "<xml>content</xml>";
        String expected = "SCANNER PARAMETERS:\n" + xml;
        String result = EWrapperMsgGenerator.scannerParameters(xml);
        assertEquals(expected, result);
    }
}
