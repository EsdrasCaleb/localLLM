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

public class EWrapperMsgGenerator_tickGeneric_3_1_Test {

    @Test
    public void testTickGeneric() throws Exception {
        // Arrange
        int tickerId = 123;
        int tickType = 2;
        double value = 100.0;
        EWrapperMsgGenerator eWrapperMsgGenerator = new EWrapperMsgGenerator();
        // Act
        String result = EWrapperMsgGenerator.tickGeneric(tickerId, tickType, value);
        // Assert
        assertEquals("id=123  TickType.LIMIT=100.0", result);
    }
}
