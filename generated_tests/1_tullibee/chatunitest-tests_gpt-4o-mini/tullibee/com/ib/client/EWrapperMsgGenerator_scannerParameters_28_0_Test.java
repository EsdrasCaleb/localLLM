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
    public void testScannerParameters_withValidXml() {
        // Arrange
        String xmlInput = "<scanner><parameter>value</parameter></scanner>";
        String expectedOutput = EWrapperMsgGenerator.SCANNER_PARAMETERS + "\n" + xmlInput;
        // Act
        String result = EWrapperMsgGenerator.scannerParameters(xmlInput);
        // Assert
        assertEquals(expectedOutput, result);
    }

    @Test
    public void testScannerParameters_withEmptyXml() {
        // Arrange
        String xmlInput = "";
        String expectedOutput = EWrapperMsgGenerator.SCANNER_PARAMETERS + "\n" + xmlInput;
        // Act
        String result = EWrapperMsgGenerator.scannerParameters(xmlInput);
        // Assert
        assertEquals(expectedOutput, result);
    }

    @Test
    public void testScannerParameters_withNullXml() {
        // Arrange
        String xmlInput = null;
        String expectedOutput = EWrapperMsgGenerator.SCANNER_PARAMETERS + "\n" + xmlInput;
        // Act
        String result = EWrapperMsgGenerator.scannerParameters(xmlInput);
        // Assert
        assertEquals(expectedOutput, result);
    }
}
