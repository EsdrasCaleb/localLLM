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
    void scannerParameters_emptyXml() {
        String xml = "";
        String expected = EWrapperMsgGenerator.SCANNER_PARAMETERS + "\n" + xml;
        String actual = EWrapperMsgGenerator.scannerParameters(xml);
        assertEquals(expected, actual);
    }

    @Test
    void scannerParameters_nonEmptyXml() {
        String xml = "<scanner_parameters>some_data</scanner_parameters>";
        String expected = EWrapperMsgGenerator.SCANNER_PARAMETERS + "\n" + xml;
        String actual = EWrapperMsgGenerator.scannerParameters(xml);
        assertEquals(expected, actual);
    }

    @Test
    void scannerParameters_nullXml() {
        String xml = null;
        String expected = EWrapperMsgGenerator.SCANNER_PARAMETERS + "\n" + xml;
        String actual = EWrapperMsgGenerator.scannerParameters(xml);
        assertEquals(expected, actual);
    }
}
