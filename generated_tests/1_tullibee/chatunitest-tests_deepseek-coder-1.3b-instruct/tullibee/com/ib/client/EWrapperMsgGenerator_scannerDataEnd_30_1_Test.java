package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

public class EWrapperMsgGenerator_scannerDataEnd_30_1_Test {

    @Test
    public void testScannerParameters() throws Exception {
        // Arrange
        EWrapperMsgGenerator eWrapperMsgGenerator = new EWrapperMsgGenerator();
        String expectedResult = "SCANNER PARAMETERS:";
        // Act
        String actualResult = (String) EWrapperMsgGenerator.class.getDeclaredField("SCANNER_PARAMETERS").get(eWrapperMsgGenerator);
        // Assert
        assertEquals(expectedResult, actualResult);
    }

    @Test
    public void testFinancialAdvisor() throws Exception {
        // Arrange
        EWrapperMsgGenerator eWrapperMsgGenerator = new EWrapperMsgGenerator();
        String expectedResult = "FA:";
        // Act
        String actualResult = (String) EWrapperMsgGenerator.class.getDeclaredField("FINANCIAL_ADVISOR").get(eWrapperMsgGenerator);
        // Assert
        assertEquals(expectedResult, actualResult);
    }
}
