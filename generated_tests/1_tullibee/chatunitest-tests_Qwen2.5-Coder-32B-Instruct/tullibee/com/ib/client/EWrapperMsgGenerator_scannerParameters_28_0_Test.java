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
        // Given
        String xml = "<scanner><param>value</param></scanner>";
        String expectedOutput = EWrapperMsgGenerator.SCANNER_PARAMETERS + "\n" + xml;
        // When
        String result = EWrapperMsgGenerator.scannerParameters(xml);
        // Then
        assertEquals(expectedOutput, result);
    }

    @Test
    public void testScannerParametersWithEmptyXml() {
        // Given
        String xml = "";
        String expectedOutput = EWrapperMsgGenerator.SCANNER_PARAMETERS + "\n";
        // When
        String result = EWrapperMsgGenerator.scannerParameters(xml);
        // Then
        assertEquals(expectedOutput, result);
    }

    @Test
    public void testScannerParametersWithNullXml() {
        // Given
        String xml = null;
        String expectedOutput = EWrapperMsgGenerator.SCANNER_PARAMETERS + "\nnull";
        // When
        String result = EWrapperMsgGenerator.scannerParameters(xml);
        // Then
        assertEquals(expectedOutput, result);
    }
}
