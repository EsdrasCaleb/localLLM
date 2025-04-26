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

public class EWrapperMsgGenerator_fundamentalData_32_0_Test {

    @Test
    public void testFundamentalData() throws Exception {
        // Arrange
        int reqId = 123;
        String data = "Sample data";
        // Act
        String result = EWrapperMsgGenerator.fundamentalData(reqId, data);
        // Assert
        assertEquals("id  = 123 len = 8\nSample data", result);
    }
}
