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

public class EWrapperMsgGenerator_scannerParameters_28_0_Test {

    @Test
    public void testScannerParameters() {
        String xml = "testXml";
        String expected = EWrapperMsgGenerator.SCANNER_PARAMETERS + "\n" + xml;
        String result = EWrapperMsgGenerator.scannerParameters(xml);
        assertEquals(expected, result);
    }
}
