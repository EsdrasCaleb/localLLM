// EWrapperMsgGenerator_tickSize_1_1_Test.java
package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Method;
import java.util.EnumSet;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class EWrapperMsgGenerator_tickSize_1_1_Test {

    @InjectMocks
    private EWrapperMsgGenerator eWrapperMsgGenerator;

    @Test
    public void testTickSize() throws Exception {
        // Arrange
        int tickerId = 1;
        int field = 1;
        int size = 10;
        // Act
        String result = (String) eWrapperMsgGenerator.getClass().getMethod("tickSize", int.class, int.class, int.class).invoke(eWrapperMsgGenerator, tickerId, field, size);
        // Assert
        assertEquals("id=1  LOT_SIZE=" + size, result);
    }

    @Test
    public void testTickSize_LotSizeField() throws Exception {
        // Arrange
        int tickerId = 1;
        int field = 2;
        int size = 10;
        // Act
        String result = (String) eWrapperMsgGenerator.getClass().getMethod("tickSize", int.class, int.class, int.class).invoke(eWrapperMsgGenerator, tickerId, field, size);
        // Assert
        assertEquals("id=1  FA=" + size, result);
    }

    @Test
    public void testTickSize_LotSizeFieldNotPresent() throws Exception {
        // Arrange
        int tickerId = 1;
        int field = 3;
        int size = 10;
        // Act
        String result = (String) eWrapperMsgGenerator.getClass().getMethod("tickSize", int.class, int.class, int.class).invoke(eWrapperMsgGenerator, tickerId, field, size);
        // Assert
        assertEquals("id=1  NOT_PRESENT=" + size, result);
    }

    @Test
    public void testTickSize_FinancialAdvisorField() throws Exception {
        // Arrange
        int tickerId = 1;
        int field = 1;
        int size = 10;
        // Act
        String result = (String) eWrapperMsgGenerator.getClass().getMethod("tickSize", int.class, int.class, int.class).invoke(eWrapperMsgGenerator, tickerId, field, size);
        // Assert
        assertEquals("id=1  LOT_SIZE=" + size, result);
    }

    @Test
    public void testTickSize_FinancialAdvisorFieldNotPresent() throws Exception {
        // Arrange
        int tickerId = 1;
        int field = 3;
        int size = 10;
        // Act
        String result = (String) eWrapperMsgGenerator.getClass().getMethod("tickSize", int.class, int.class, int.class).invoke(eWrapperMsgGenerator, tickerId, field, size);
        // Assert
        assertEquals("id=1  NOT_PRESENT=" + size, result);
    }
}
