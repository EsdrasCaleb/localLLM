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

public class EWrapperMsgGenerator_contractDetailsEnd_18_0_Test {

    @Test
    public void testEWrapperMsgGenerator() throws Exception {
        // Arrange
        EWrapperMsgGenerator eWrapperMsgGenerator = new EWrapperMsgGenerator();
        String expectedResult = "SCANNER PARAMETERS:";
        // Act
        Field field = EWrapperMsgGenerator.class.getDeclaredField("SCANNER_PARAMETERS");
        field.setAccessible(true);
        String result = (String) field.get(eWrapperMsgGenerator);
        // Assert
        assertEquals(expectedResult, result);
    }
}
