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

public class EWrapperMsgGenerator_fundamentalData_32_0_Test {

    @Test
    public void testFundamentalData() throws Exception {
        // Arrange
        int reqId = 123;
        String data = "TestData";
        EWrapperMsgGenerator eWrapperMsgGenerator = new EWrapperMsgGenerator();
        Field field = EWrapperMsgGenerator.class.getDeclaredField("SCANNER_PARAMETERS");
        field.setAccessible(true);
        String expected = "id  = " + reqId + " len = " + data.length() + '\n' + data;
        // Act
        String result = (String) field.get(eWrapperMsgGenerator);
        // Assert
        assertEquals(expected, result);
    }
}
